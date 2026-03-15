/***************************************************************************/
/* Copyright(C) 2021
 *
 * QuadWild-BiMDF-CUDA
 * CUDA-accelerated version with pipeline checkpointing.
 *
 * Original: Reliable Feature-Line Driven Quad-Remeshing (Siggraph 2021)
 * CUDA additions: Pipeline checkpoint system, GPU smoothing, GPU chart data
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 ****************************************************************************/

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/polygonal_algorithms.h>
#include <vcg/complex/algorithms/update/topology.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <stdio.h>

#include "load_save.h"
#include "mesh_types.h"

#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>

#include "smooth_mesh.h"
#include "quad_from_patches.h"

#include <clocale>
#include <cstring>

#include <libTimekeeper/StopWatch.hh>
#include <libTimekeeper/StopWatchPrinting.hh>

#include <nlohmann/json.hpp>
#include <libsatsuma/Extra/json.hh>
#include <libTimekeeper/json.hh>
#include <quadretopology/qr_eval_quantization_json.h>

#include <iostream>

// Checkpoint system (always available)
#include <qw_checkpoint.h>
#include <qw_serialize.h>

// CUDA kernels + VCG bridge (only with CUDA)
#if QUADWILD_HAS_CUDA
#include <vcg_bridge.h>
#include <smooth_kernels.h>
#include <chart_kernels.h>
#endif

#ifdef _WIN32
#  include <windows.h>
#  include <stdlib.h>
#  include <errhandlingapi.h>
#endif

using HSW = Timekeeper::HierarchicalStopWatch;
using Timekeeper::ScopedStopWatch;

bool LocalUVSm=false;
typename TriangleMesh::ScalarType avgEdge(const TriangleMesh& trimesh);
void loadSetupFile(const std::string& path, QuadRetopology::Parameters& parameters, float& scaleFactor, int& fixedChartClusters);
void SaveSetupFile(const std::string& path, QuadRetopology::Parameters& parameters, float& scaleFactor, int& fixedChartClusters);

// ============================================================
// Helper: Extract CheckpointData from live state
// ============================================================

static void extract_trimesh_data(const TriangleMesh& mesh, qw::CheckpointData& cd) {
    cd.tri_verts.resize(mesh.vert.size() * 3);
    for (size_t i = 0; i < mesh.vert.size(); ++i) {
        cd.tri_verts[i*3+0] = (float)mesh.vert[i].cP()[0];
        cd.tri_verts[i*3+1] = (float)mesh.vert[i].cP()[1];
        cd.tri_verts[i*3+2] = (float)mesh.vert[i].cP()[2];
    }
    cd.tri_faces.resize(mesh.face.size() * 3);
    for (size_t i = 0; i < mesh.face.size(); ++i) {
        for (int j = 0; j < 3; ++j)
            cd.tri_faces[i*3+j] = (int)(mesh.face[i].cV(j) - &mesh.vert[0]);
    }
}

static void extract_chartdata(const QuadRetopology::ChartData& chartData, qw::CheckpointData& cd) {
    cd.has_chartdata = true;
    int nc = (int)chartData.charts.size();

    cd.chart_num_sides.resize(nc);
    cd.chart_labels.resize(nc);
    cd.chart_faces.resize(nc);
    cd.chart_border_faces.resize(nc);
    cd.chart_adjacent.resize(nc);
    cd.side_vertices.resize(nc);
    cd.side_subsides.resize(nc);
    cd.side_reversed.resize(nc);
    cd.side_lengths.resize(nc);
    cd.side_sizes.resize(nc);
    cd.chart_subsides.resize(nc);

    for (int c = 0; c < nc; ++c) {
        auto& chart = chartData.charts[c];
        cd.chart_num_sides[c] = (int)chart.chartSides.size();
        cd.chart_labels[c] = chart.label;
        cd.chart_faces[c] = chart.faces;
        cd.chart_border_faces[c] = chart.borderFaces;
        cd.chart_adjacent[c] = chart.adjacentCharts;
        cd.chart_subsides[c] = chart.chartSubsides;

        int ns = (int)chart.chartSides.size();
        cd.side_vertices[c].resize(ns);
        cd.side_subsides[c].resize(ns);
        cd.side_reversed[c].resize(ns);
        cd.side_lengths[c].resize(ns);
        cd.side_sizes[c].resize(ns);

        for (int s = 0; s < ns; ++s) {
            cd.side_vertices[c][s] = chart.chartSides[s].vertices;
            cd.side_subsides[c][s] = chart.chartSides[s].subsides;
            cd.side_reversed[c][s] = chart.chartSides[s].reversedSubside;
            cd.side_lengths[c][s] = chart.chartSides[s].length;
            cd.side_sizes[c][s] = chart.chartSides[s].size;
        }
    }

    int nsub = (int)chartData.subsides.size();
    cd.subside_incident_charts.resize(nsub);
    cd.subside_incident_chart_subside_id.resize(nsub);
    cd.subside_incident_chart_side_id.resize(nsub);
    cd.subside_vertices.resize(nsub);
    cd.subside_lengths.resize(nsub);
    cd.subside_sizes.resize(nsub);
    cd.subside_is_on_border.resize(nsub);

    for (int i = 0; i < nsub; ++i) {
        auto& sub = chartData.subsides[i];
        cd.subside_incident_charts[i] = {sub.incidentCharts[0], sub.incidentCharts[1]};
        cd.subside_incident_chart_subside_id[i] = {sub.incidentChartSubsideId[0], sub.incidentChartSubsideId[1]};
        cd.subside_incident_chart_side_id[i] = {sub.incidentChartSideId[0], sub.incidentChartSideId[1]};
        cd.subside_vertices[i] = sub.vertices;
        cd.subside_lengths[i] = sub.length;
        cd.subside_sizes[i] = sub.size;
        cd.subside_is_on_border[i] = sub.isOnBorder;
    }

    cd.label_set.clear();
    for (int l : chartData.labels) cd.label_set.push_back(l);
}

static void restore_chartdata(const qw::CheckpointData& cd, QuadRetopology::ChartData& chartData) {
    int nc = (int)cd.chart_faces.size();
    chartData.charts.resize(nc);

    for (int c = 0; c < nc; ++c) {
        auto& chart = chartData.charts[c];
        chart.label = cd.chart_labels[c];
        chart.faces = cd.chart_faces[c];
        chart.borderFaces = cd.chart_border_faces[c];
        chart.adjacentCharts = cd.chart_adjacent[c];
        chart.chartSubsides = cd.chart_subsides[c];

        int ns = cd.chart_num_sides[c];
        chart.chartSides.resize(ns);
        for (int s = 0; s < ns; ++s) {
            chart.chartSides[s].vertices = cd.side_vertices[c][s];
            chart.chartSides[s].subsides = cd.side_subsides[c][s];
            chart.chartSides[s].reversedSubside = cd.side_reversed[c][s];
            chart.chartSides[s].length = cd.side_lengths[c][s];
            chart.chartSides[s].size = cd.side_sizes[c][s];
        }
    }

    int nsub = (int)cd.subside_incident_charts.size();
    chartData.subsides.resize(nsub);
    for (int i = 0; i < nsub; ++i) {
        auto& sub = chartData.subsides[i];
        sub.incidentCharts = {cd.subside_incident_charts[i][0], cd.subside_incident_charts[i][1]};
        sub.incidentChartSubsideId = {cd.subside_incident_chart_subside_id[i][0], cd.subside_incident_chart_subside_id[i][1]};
        sub.incidentChartSideId = {cd.subside_incident_chart_side_id[i][0], cd.subside_incident_chart_side_id[i][1]};
        sub.vertices = cd.subside_vertices[i];
        sub.length = cd.subside_lengths[i];
        sub.size = cd.subside_sizes[i];
        sub.isOnBorder = cd.subside_is_on_border[i];
    }

    chartData.labels.clear();
    for (int l : cd.label_set) chartData.labels.insert(l);
}

// ============================================================
// Parse checkpoint CLI flags from argc/argv
// Returns index of first non-checkpoint argument consumed
// ============================================================

struct CheckpointFlags {
    std::string save_dir;
    qw::PipelineStage save_at = qw::STAGE_NONE;
    bool save_all = false;
    qw::PipelineStage run_from = qw::STAGE_NONE;
    qw::PipelineStage run_to = qw::STAGE_NONE;
    bool list_stages = false;
    bool use_cuda_smooth = true;
    int flow_strategy = 9;  // 0=satsuma ... 7=suitor 8=pdhg-v2 9=hybrid
};

static void parse_checkpoint_flags(int argc, char* argv[], CheckpointFlags& flags) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-save-dir") == 0 && i + 1 < argc) {
            flags.save_dir = argv[++i];
        } else if (strcmp(argv[i], "-save-at") == 0 && i + 1 < argc) {
            flags.save_at = qw::stage_from_name(argv[++i]);
        } else if (strcmp(argv[i], "-save-all") == 0) {
            flags.save_all = true;
        } else if (strcmp(argv[i], "-run-from") == 0 && i + 1 < argc) {
            flags.run_from = qw::stage_from_name(argv[++i]);
        } else if (strcmp(argv[i], "-run-to") == 0 && i + 1 < argc) {
            flags.run_to = qw::stage_from_name(argv[++i]);
        } else if (strcmp(argv[i], "-list-stages") == 0) {
            flags.list_stages = true;
        } else if (strcmp(argv[i], "-no-cuda-smooth") == 0) {
            flags.use_cuda_smooth = false;
        } else if (strcmp(argv[i], "-flow-strategy") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "satsuma") == 0) flags.flow_strategy = 0;
            else if (strcmp(argv[i], "early-term") == 0) flags.flow_strategy = 1;
            else if (strcmp(argv[i], "admm") == 0) flags.flow_strategy = 2;
            else if (strcmp(argv[i], "phase1-only") == 0) flags.flow_strategy = 3;
            else if (strcmp(argv[i], "pdhg") == 0) flags.flow_strategy = 4;
            else if (strcmp(argv[i], "pdhg-direct") == 0) flags.flow_strategy = 5;
            else if (strcmp(argv[i], "sa") == 0) flags.flow_strategy = 6;
            else if (strcmp(argv[i], "suitor") == 0) flags.flow_strategy = 7;
            else if (strcmp(argv[i], "pdhg-v2") == 0) flags.flow_strategy = 8;
            else if (strcmp(argv[i], "hybrid") == 0) flags.flow_strategy = 9;
            else if (strcmp(argv[i], "sinpen") == 0) flags.flow_strategy = 10;
            else if (strcmp(argv[i], "pump") == 0) flags.flow_strategy = 11;
            else if (strcmp(argv[i], "suitor-aug") == 0) flags.flow_strategy = 12;
            else if (strcmp(argv[i], "adaptive") == 0) flags.flow_strategy = 13;
            else if (strcmp(argv[i], "directed") == 0) flags.flow_strategy = 14;
            else std::cerr << "Unknown flow strategy: " << argv[i] << std::endl;
        }
    }
}

// ============================================================
// Main
// ============================================================

int actual_main(int argc, char *argv[])
{
#ifdef _WIN32
    SetErrorMode(0);
#endif
    HSW sw_root("main");
    HSW sw_load("load", sw_root);
    HSW sw_chartdata("chartdata", sw_root);
    HSW sw_smooth("smooth", sw_root);
    HSW sw_save("save", sw_root);

    sw_root.resume();

    std::setlocale(LC_NUMERIC, "en_US.UTF-8");

    // Parse checkpoint flags
    CheckpointFlags ckpt;
    parse_checkpoint_flags(argc, argv, ckpt);

    if (ckpt.list_stages) {
        qw::list_stages();
        return 0;
    }

    int CurrNum=0;

    TriangleMesh trimesh;
    std::vector<std::vector<size_t>> trimeshPartitions;
    std::vector<std::vector<size_t>> trimeshCorners;
    std::vector<std::pair<size_t,size_t> > trimeshFeatures;
    std::vector<size_t> trimeshFeaturesC;

    PolyMesh quadmesh;
    std::vector<std::vector<size_t>> quadmeshPartitions;
    std::vector<std::vector<size_t>> quadmeshCorners;
    std::vector<int> ilpResult;

    if(argc<2)
    {
        std::cerr << "usage: " << argv[0] << " <input.obj> [num] [setup.txt] [out_stats.json]"
                  << " [-save-dir DIR] [-save-at STAGE] [-save-all]"
                  << " [-run-from STAGE] [-run-to STAGE] [-list-stages]"
                  << " [-no-cuda-smooth]"
                  << std::endl;
        exit(1);
    }

    // Find positional args (skip checkpoint flags)
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-save-dir") == 0 ||
            strcmp(argv[i], "-save-at") == 0 ||
            strcmp(argv[i], "-run-from") == 0 ||
            strcmp(argv[i], "-run-to") == 0 ||
            strcmp(argv[i], "-flow-strategy") == 0) {
            ++i; // skip value
            continue;
        }
        if (strcmp(argv[i], "-save-all") == 0 ||
            strcmp(argv[i], "-list-stages") == 0 ||
            strcmp(argv[i], "-no-cuda-smooth") == 0) {
            continue;
        }
        pos_args.push_back(argv[i]);
    }

    if (pos_args.empty()) {
        std::cerr << "ERROR: no input mesh specified" << std::endl;
        return 1;
    }

    if (pos_args.size() > 1) CurrNum = atoi(pos_args[1].c_str());

    std::string configFilename = "basic_setup.txt";
    if (pos_args.size() > 2) configFilename = pos_args[2];

    std::string json_filename = "";
    if (pos_args.size() > 3) json_filename = pos_args[3];

    QuadRetopology::Parameters parameters;
    float scaleFactor;
    int fixedChartClusters;

    // ============================================================
    // Pipeline control lambdas (like QuadriFlow-cuda)
    // ============================================================

    auto should_run = [&](qw::PipelineStage stage) -> bool {
        if (ckpt.run_from != qw::STAGE_NONE && stage <= ckpt.run_from) return false;
        if (ckpt.run_to != qw::STAGE_NONE && stage > ckpt.run_to) return false;
        return true;
    };

    auto maybe_save = [&](qw::PipelineStage stage, const qw::CheckpointData& cd, const qw::CheckpointHeader& hdr) {
        if (!ckpt.save_dir.empty() && (ckpt.save_all || stage == ckpt.save_at)) {
            qw::save_checkpoint(cd, stage, ckpt.save_dir.c_str(), hdr);
        }
    };

    auto should_stop = [&](qw::PipelineStage stage) -> bool {
        return (ckpt.run_to != qw::STAGE_NONE && stage >= ckpt.run_to);
    };

    // State tracking
    qw::CheckpointData cd;
    QuadRetopology::ChartData chartData;
    double edgeSize = 0;
    std::vector<double> edgeFactor;

    // ============================================================
    // Load from checkpoint or from files
    // ============================================================

    if (ckpt.run_from != qw::STAGE_NONE) {
        if (ckpt.save_dir.empty()) {
            std::cerr << "ERROR: -run-from requires -save-dir" << std::endl;
            return 1;
        }
        qw::PipelineStage loaded = qw::load_checkpoint(cd, ckpt.save_dir.c_str(), ckpt.run_from);
        if (loaded == qw::STAGE_NONE) {
            std::cerr << "ERROR: Failed to load checkpoint for stage '"
                      << qw::stage_name(ckpt.run_from) << "'" << std::endl;
            return 1;
        }

        // Reconstruct VCG mesh from checkpoint data
        // (For simplicity, reload from the original file specified in args)
        // The checkpoint data is used for ChartData/ILP results/etc.
    }

    // ============================================================
    // STAGE: post-load (load mesh + partitions + corners + features)
    // ============================================================

    if (should_run(qw::STAGE_POST_LOAD)) {
        sw_load.resume();
        loadSetupFile(configFilename, parameters, scaleFactor, fixedChartClusters);

        parameters.chartSmoothingIterations = 0;
        parameters.quadrangulationFixedSmoothingIterations = 0;
        parameters.quadrangulationNonFixedSmoothingIterations = 0;
        parameters.feasibilityFix = false;
        parameters.cuda_flow_strategy = ckpt.flow_strategy;

        std::string meshFilename = pos_args[0];
        int mask;
        vcg::tri::io::ImporterOBJ<TriangleMesh>::LoadMask(meshFilename.c_str(), mask);
        int err = vcg::tri::io::ImporterOBJ<TriangleMesh>::Open(trimesh, meshFilename.c_str(), mask);

        if ((err!=0)&&(err!=5)) {
            std::cout<<"Error loading mesh from file " << meshFilename <<std::endl;
            exit(0);
        }
        std::cout<<"MESH NAME "<<meshFilename.c_str()<<std::endl;
        std::cout<<"Loaded "<<trimesh.vert.size()<<" vertices"<<std::endl;
        std::cout<<"Loaded "<<trimesh.face.size()<<" faces"<<std::endl;

        std::string partitionFilename = meshFilename;
        partitionFilename.erase(partitionFilename.find_last_of("."));
        partitionFilename.append(".patch");
        trimeshPartitions = loadPatches(partitionFilename);
        std::cout<<"Loaded "<<trimeshPartitions.size()<<" patches"<<std::endl;

        std::string cornerFilename = meshFilename;
        cornerFilename.erase(cornerFilename.find_last_of("."));
        cornerFilename.append(".corners");
        trimeshCorners = loadCorners(cornerFilename);
        std::cout<<"Loaded "<<trimeshCorners.size()<<" corners set"<<std::endl;

        std::string featureFilename = meshFilename;
        featureFilename.erase(featureFilename.find_last_of("."));
        featureFilename.append(".feature");
        trimeshFeatures = LoadFeatures(featureFilename);
        std::cout<<"Loaded "<<trimeshFeatures.size()<<" features"<<std::endl;

        std::string featureCFilename = meshFilename;
        featureCFilename.erase(featureCFilename.find_last_of("."));
        featureCFilename.append(".c_feature");
        trimeshFeaturesC = loadFeatureCorners(featureCFilename);
        std::cout<<"Loaded "<<trimeshFeaturesC.size()<<" corner features"<<std::endl;
        loadFeatureCorners(featureCFilename);

        OrientIfNeeded(trimesh,trimeshPartitions,trimeshCorners,trimeshFeatures,trimeshFeaturesC);

        QuadRetopology::internal::updateAllMeshAttributes(trimesh);
        edgeSize = avgEdge(trimesh) * scaleFactor;
        std::cout<<"Edge Size "<<edgeSize<<std::endl;
        edgeFactor.assign(trimeshPartitions.size(), edgeSize);

        sw_load.stop();

        // Save checkpoint
        extract_trimesh_data(trimesh, cd);
        cd.partitions = trimeshPartitions;
        cd.corners = trimeshCorners;
        cd.features = trimeshFeatures;
        cd.feature_corners = trimeshFeaturesC;
        cd.edge_size = edgeSize;

        qw::CheckpointHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        strncpy(hdr.input_mesh, pos_args[0].c_str(), sizeof(hdr.input_mesh) - 1);
        strncpy(hdr.setup_file, configFilename.c_str(), sizeof(hdr.setup_file) - 1);
        hdr.alpha = parameters.alpha;
        hdr.scale_factor = scaleFactor;
        hdr.fixed_chart_clusters = fixedChartClusters;
        hdr.use_flow_solver = parameters.useFlowSolver ? 1 : 0;
        hdr.num_vertices = (int)trimesh.vert.size();
        hdr.num_faces = (int)trimesh.face.size();
        hdr.num_patches = (int)trimeshPartitions.size();

        maybe_save(qw::STAGE_POST_LOAD, cd, hdr);
        if (should_stop(qw::STAGE_POST_LOAD)) goto done;
    } else {
        // Load mesh anyway (needed for subsequent stages)
        loadSetupFile(configFilename, parameters, scaleFactor, fixedChartClusters);
        parameters.chartSmoothingIterations = 0;
        parameters.quadrangulationFixedSmoothingIterations = 0;
        parameters.quadrangulationNonFixedSmoothingIterations = 0;
        parameters.feasibilityFix = false;
        parameters.cuda_flow_strategy = ckpt.flow_strategy;

        std::string meshFilename = pos_args[0];
        int mask;
        vcg::tri::io::ImporterOBJ<TriangleMesh>::LoadMask(meshFilename.c_str(), mask);
        vcg::tri::io::ImporterOBJ<TriangleMesh>::Open(trimesh, meshFilename.c_str(), mask);

        std::string base = meshFilename;
        base.erase(base.find_last_of("."));
        trimeshPartitions = loadPatches(base + ".patch");
        trimeshCorners = loadCorners(base + ".corners");
        trimeshFeatures = LoadFeatures(base + ".feature");
        trimeshFeaturesC = loadFeatureCorners(base + ".c_feature");
        loadFeatureCorners(base + ".c_feature");
        OrientIfNeeded(trimesh,trimeshPartitions,trimeshCorners,trimeshFeatures,trimeshFeaturesC);
        QuadRetopology::internal::updateAllMeshAttributes(trimesh);
        edgeSize = avgEdge(trimesh) * scaleFactor;
        edgeFactor.assign(trimeshPartitions.size(), edgeSize);
    }

    // ============================================================
    // STAGE: post-chartdata (compute chart data)
    // ============================================================

    if (should_run(qw::STAGE_POST_CHARTDATA)) {
        sw_chartdata.resume();
        std::cout << "Computing chart data..." << std::endl;

        if (ckpt.run_from >= qw::STAGE_POST_CHARTDATA && cd.has_chartdata) {
            // Restore from checkpoint
            restore_chartdata(cd, chartData);
            std::cout << "Restored chart data from checkpoint: "
                      << chartData.charts.size() << " charts, "
                      << chartData.subsides.size() << " subsides" << std::endl;
        } else {
            chartData = QuadRetopology::computeChartData(
                trimesh, trimeshPartitions, trimeshCorners);
        }
        sw_chartdata.stop();

        ilpResult.clear();
        ilpResult.resize(chartData.subsides.size(), ILP_FIND_SUBDIVISION);

        // Save checkpoint
        extract_chartdata(chartData, cd);

        qw::CheckpointHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        strncpy(hdr.input_mesh, pos_args[0].c_str(), sizeof(hdr.input_mesh) - 1);
        hdr.alpha = parameters.alpha;
        hdr.scale_factor = scaleFactor;
        hdr.fixed_chart_clusters = fixedChartClusters;
        hdr.use_flow_solver = parameters.useFlowSolver ? 1 : 0;
        hdr.num_vertices = (int)trimesh.vert.size();
        hdr.num_faces = (int)trimesh.face.size();
        hdr.num_patches = (int)trimeshPartitions.size();
        hdr.num_subsides = (int)chartData.subsides.size();

        maybe_save(qw::STAGE_POST_CHARTDATA, cd, hdr);
        if (should_stop(qw::STAGE_POST_CHARTDATA)) goto done;
    } else if (cd.has_chartdata) {
        restore_chartdata(cd, chartData);
        ilpResult.clear();
        ilpResult.resize(chartData.subsides.size(), ILP_FIND_SUBDIVISION);
    }

    // ============================================================
    // STAGE: post-flow (solve BiMDF / ILP)
    // ============================================================

    {
        std::vector<Satsuma::BiMDFFullResult> bimdf_results;
        std::vector<QuadRetopology::FlowStats> flow_stats;
        std::vector<std::vector<QuadRetopology::ILPStats>> ilp_stats_per_cluster;

        if (should_run(qw::STAGE_POST_FLOW)) {
            if (ckpt.run_from >= qw::STAGE_POST_FLOW && cd.has_flow_result) {
                // Restore ILP result from checkpoint
                ilpResult.resize(cd.ilp_result.size());
                for (size_t i = 0; i < cd.ilp_result.size(); ++i)
                    ilpResult[i] = cd.ilp_result[i];
                std::cout << "Restored flow result from checkpoint" << std::endl;
            } else {
                std::cout << "Solving quantization (BiMDF/ILP)..." << std::endl;
                double gap;
                auto subdiv_res = QuadRetopology::findSubdivisions(
                    chartData, edgeFactor, parameters, gap, ilpResult);
                bimdf_results = std::move(subdiv_res.bimdf_results);
                flow_stats = std::move(subdiv_res.flow_stats);
            }

            // Save checkpoint
            cd.has_flow_result = true;
            cd.ilp_result.resize(ilpResult.size());
            for (size_t i = 0; i < ilpResult.size(); ++i)
                cd.ilp_result[i] = ilpResult[i];

            qw::CheckpointHeader hdr;
            memset(&hdr, 0, sizeof(hdr));
            strncpy(hdr.input_mesh, pos_args[0].c_str(), sizeof(hdr.input_mesh) - 1);
            hdr.alpha = parameters.alpha;
            hdr.scale_factor = scaleFactor;
            hdr.fixed_chart_clusters = fixedChartClusters;
            hdr.use_flow_solver = parameters.useFlowSolver ? 1 : 0;
            hdr.num_vertices = (int)trimesh.vert.size();
            hdr.num_faces = (int)trimesh.face.size();
            hdr.num_patches = (int)trimeshPartitions.size();
            hdr.num_subsides = (int)chartData.subsides.size();

            maybe_save(qw::STAGE_POST_FLOW, cd, hdr);
            if (should_stop(qw::STAGE_POST_FLOW)) goto done;
        } else if (cd.has_flow_result) {
            ilpResult.resize(cd.ilp_result.size());
            for (size_t i = 0; i < cd.ilp_result.size(); ++i)
                ilpResult[i] = cd.ilp_result[i];
        }

    // ============================================================
    // STAGE: post-quadrangulate
    // ============================================================

    if (should_run(qw::STAGE_POST_QUADRANGULATE)) {
        std::cout << "Quadrangulating..." << std::endl;

        auto quant_eval = QuadRetopology::evaluate_quantization(
            chartData, edgeFactor, parameters, ilpResult);
        std::cout << "quantisation evaluation results: \n " << quant_eval << std::endl;

        std::vector<size_t> fixedPositionSubsides;
        std::vector<int> quadmeshLabel;
        QuadRetopology::quadrangulate(
            trimesh, chartData, fixedPositionSubsides,
            ilpResult, parameters, quadmesh,
            quadmeshLabel, quadmeshPartitions, quadmeshCorners);

        // Save unsmoothed output
        std::string meshFilename = pos_args[0];
        std::string outputFilename = meshFilename;
        outputFilename.erase(outputFilename.find_last_of("."));
        outputFilename += std::string("_") + std::to_string(CurrNum)
                       + std::string("_quadrangulation.obj");

        vcg::tri::io::ExporterOBJ<PolyMesh>::Save(quadmesh, outputFilename.c_str(),
            vcg::tri::io::Mask::IOM_FACECOLOR);

        qw::CheckpointHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        strncpy(hdr.input_mesh, pos_args[0].c_str(), sizeof(hdr.input_mesh) - 1);
        hdr.alpha = parameters.alpha;
        hdr.num_vertices = (int)trimesh.vert.size();
        hdr.num_faces = (int)trimesh.face.size();
        hdr.num_patches = (int)trimeshPartitions.size();

        maybe_save(qw::STAGE_POST_QUADRANGULATE, cd, hdr);
        if (should_stop(qw::STAGE_POST_QUADRANGULATE)) goto done;
    }

    // ============================================================
    // STAGE: post-smooth (final smoothing - CUDA accelerated)
    // ============================================================

    if (should_run(qw::STAGE_POST_SMOOTH)) {
        sw_smooth.resume();

        std::vector<size_t> QuadPart(quadmesh.face.size(),0);
        for (size_t i=0;i<quadmeshPartitions.size();i++)
            for (size_t j=0;j<quadmeshPartitions[i].size();j++)
                QuadPart[quadmeshPartitions[i][j]]=i;

        std::vector<size_t> TriPart(trimesh.face.size(),0);
        for (size_t i=0;i<trimeshPartitions.size();i++)
            for (size_t j=0;j<trimeshPartitions[i].size();j++)
                TriPart[trimeshPartitions[i][j]]=i;

        std::vector<size_t> QuadCornersVect;
        for (size_t i=0;i<quadmeshCorners.size();i++)
            for (size_t j=0;j<quadmeshCorners[i].size();j++)
                QuadCornersVect.push_back(quadmeshCorners[i][j]);
        std::sort(QuadCornersVect.begin(),QuadCornersVect.end());
        auto last=std::unique(QuadCornersVect.begin(),QuadCornersVect.end());
        QuadCornersVect.erase(last, QuadCornersVect.end());

        std::cout<<"** SMOOTHING **"<<std::endl;

        // Use CUDA smoothing if available, else fall back to CPU
        bool cuda_success = false;
#if QUADWILD_HAS_CUDA
        if (ckpt.use_cuda_smooth) {
            try {
                std::cout << "Using CUDA-accelerated smoothing..." << std::endl;

                // Extract flat arrays from VCG meshes
                std::vector<float> poly_verts;
                qw::bridge::extract_vertex_positions(quadmesh, poly_verts);

                std::vector<int> adj_offsets, adj_indices;
                qw::bridge::build_vertex_adjacency_csr(quadmesh, adj_offsets, adj_indices);

                std::vector<float> tri_verts;
                qw::bridge::extract_vertex_positions(trimesh, tri_verts);
                std::vector<int> tri_faces;
                qw::bridge::extract_tri_faces(trimesh, tri_faces);

                // Build projection type array
                // For now, all vertices are surface-projected (simplified)
                // A full implementation would use GetProjectionBasis
                std::vector<int> proj_type(quadmesh.vert.size(), qw::cuda::PROJ_SURFACE);
                for (size_t c : QuadCornersVect) {
                    if (c < proj_type.size()) proj_type[c] = qw::cuda::PROJ_CORNER;
                }

                // Build edge mesh data for sharp feature projection
                std::vector<float> edge_verts;
                std::vector<int> sharp_vert_group;
                std::vector<int> edge_group_offsets;
                std::vector<int> edge_group_indices;
                qw::bridge::build_sharp_feature_data(
                    quadmesh, trimesh, trimeshFeatures,
                    TriPart, QuadPart,
                    edge_verts, sharp_vert_group,
                    edge_group_offsets, edge_group_indices);

                int num_edge_groups = edge_group_offsets.empty() ? 0 : (int)edge_group_offsets.size() - 1;
                int num_edge_group_entries = (int)edge_group_indices.size();

                // Initialize CUDA smoothing
                float cell_size = (float)edgeSize * 3.0f;
                qw::cuda::SmoothData sd;
                qw::cuda::cuda_smooth_init(
                    sd,
                    poly_verts.data(), (int)quadmesh.vert.size(),
                    adj_offsets.data(), adj_indices.data(), (int)adj_indices.size(),
                    proj_type.data(),
                    tri_verts.data(), (int)trimesh.vert.size(),
                    tri_faces.data(), (int)trimesh.face.size(),
                    edge_verts.empty() ? nullptr : edge_verts.data(),
                    (int)(edge_verts.size() / 3),
                    sharp_vert_group.empty() ? nullptr : sharp_vert_group.data(),
                    edge_group_offsets.empty() ? nullptr : edge_group_offsets.data(),
                    edge_group_indices.empty() ? nullptr : edge_group_indices.data(),
                    num_edge_groups, num_edge_group_entries,
                    cell_size);

                // Run 30 iterations with 1 back-projection step
                qw::cuda::cuda_smooth(sd, 30, 1, 0.5f);

                // Download results
                qw::cuda::cuda_smooth_download(sd, poly_verts.data());
                qw::cuda::cuda_smooth_destroy(sd);

                // Write back to VCG mesh
                qw::bridge::write_vertex_positions(quadmesh, poly_verts);

                cuda_success = true;
                std::cout << "CUDA smoothing complete." << std::endl;
            } catch (...) {
                std::cerr << "CUDA smoothing failed, falling back to CPU" << std::endl;
            }
        }
#endif // QUADWILD_HAS_CUDA

        if (!cuda_success) {
            // CPU fallback
            if (LocalUVSm)
                LocalUVSmooth(quadmesh,trimesh,trimeshFeatures,trimeshFeaturesC,30);
            else
                MultiCostraintSmooth(quadmesh,trimesh,trimeshFeatures,trimeshFeaturesC,
                    TriPart,QuadCornersVect,QuadPart,0.5,edgeSize,30,1);
        }

        sw_smooth.stop();

        // Save smoothed output
        sw_save.resume();
        std::string meshFilename = pos_args[0];
        std::string outputFilename = meshFilename;
        outputFilename.erase(outputFilename.find_last_of("."));
        outputFilename += std::string("_") + std::to_string(CurrNum)
                       + std::string("_quadrangulation_smooth.obj");
        vcg::tri::io::ExporterOBJ<PolyMesh>::Save(quadmesh, outputFilename.c_str(),
            vcg::tri::io::Mask::IOM_FACECOLOR);
        sw_save.stop();
    }

    } // end bimdf_results scope

done:
    sw_root.stop();
    auto sw_result = Timekeeper::HierarchicalStopWatchResult(sw_root);
    std::cout << "\n" << sw_result << std::endl;

    if (!json_filename.empty()) {
        auto json = nlohmann::json{{"runtimes", sw_result}};
        std::ofstream json_file{json_filename};
        json_file << std::setw(4) << json;
    }
    return 0;
}


typename TriangleMesh::ScalarType avgEdge(const TriangleMesh& trimesh)
{
    typedef typename TriangleMesh::ScalarType ScalarType;
    ScalarType AvgVal=0;
    size_t Num=0;
    for (size_t i=0;i<trimesh.face.size();i++)
        for (size_t j=0;j<3;j++)
        {
            AvgVal+=(trimesh.face[i].cP0(j)-trimesh.face[i].cP1(j)).Norm();
            Num++;
        }
    return (AvgVal/Num);
}

void loadSetupFile(const std::string& path, QuadRetopology::Parameters& parameters, float& scaleFactor, int& fixedChartClusters)
{
    FILE *f=fopen(path.c_str(),"rt");
    if (f == nullptr) {
        throw std::runtime_error(std::string("cannot open setup file ") + path);
    }

    float alphaF;
    fscanf(f,"alpha %f\n",&alphaF);
    std::cout<<"ALPHA "<<alphaF<<std::endl;
    parameters.alpha=alphaF;

    int IntVar=0;
    fscanf(f,"ilpMethod %d\n",&IntVar);
    if (IntVar==0)
        parameters.ilpMethod=QuadRetopology::ILPMethod::ABS;
    else
        parameters.ilpMethod=QuadRetopology::ILPMethod::LEASTSQUARES;

    float limitF;
    fscanf(f,"timeLimit %f\n",&limitF);
    parameters.timeLimit=limitF;

    float gapF;
    fscanf(f,"gapLimit %f\n",&gapF);
    parameters.gapLimit=gapF;

    IntVar=0;
    fscanf(f,"callbackTimeLimit %d",&IntVar);
    parameters.callbackTimeLimit.resize(IntVar);
    for (int i = 0; i < IntVar; i++) {
        fscanf(f," %f", &parameters.callbackTimeLimit[i]);
    }
    fscanf(f,"\n");

    IntVar=0;
    fscanf(f,"callbackGapLimit %d",&IntVar);
    parameters.callbackGapLimit.resize(IntVar);
    for (int i = 0; i < IntVar; i++) {
        fscanf(f," %f", &parameters.callbackGapLimit[i]);
    }
    fscanf(f,"\n");

    float mingapF;
    fscanf(f,"minimumGap %f\n",&mingapF);
    parameters.minimumGap=mingapF;

    IntVar=0;
    fscanf(f,"isometry %d\n",&IntVar);
    parameters.isometry = (IntVar != 0);

    IntVar=0;
    fscanf(f,"regularityQuadrilaterals %d\n",&IntVar);
    parameters.regularityQuadrilaterals = (IntVar != 0);

    IntVar=0;
    fscanf(f,"regularityNonQuadrilaterals %d\n",&IntVar);
    parameters.regularityNonQuadrilaterals = (IntVar != 0);

    float regularityNonQuadrilateralsWeight;
    fscanf(f,"regularityNonQuadrilateralsWeight %f\n",&regularityNonQuadrilateralsWeight);
    parameters.regularityNonQuadrilateralsWeight=regularityNonQuadrilateralsWeight;

    IntVar=0;
    fscanf(f,"alignSingularities %d\n",&IntVar);
    parameters.alignSingularities = (IntVar != 0);

    float alignSingularitiesWeight;
    fscanf(f,"alignSingularitiesWeight %f\n",&alignSingularitiesWeight);
    parameters.alignSingularitiesWeight=alignSingularitiesWeight;

    IntVar=0;
    fscanf(f,"repeatLosingConstraintsIterations %d\n",&IntVar);
    parameters.repeatLosingConstraintsIterations=IntVar;

    IntVar=0;
    fscanf(f,"repeatLosingConstraintsQuads %d\n",&IntVar);
    parameters.repeatLosingConstraintsQuads = (IntVar != 0);

    IntVar=0;
    fscanf(f,"repeatLosingConstraintsNonQuads %d\n",&IntVar);
    parameters.repeatLosingConstraintsNonQuads = (IntVar != 0);

    IntVar=0;
    fscanf(f,"repeatLosingConstraintsAlign %d\n",&IntVar);
    parameters.repeatLosingConstraintsAlign = (IntVar != 0);

    IntVar=0;
    fscanf(f,"hardParityConstraint %d\n",&IntVar);
    parameters.hardParityConstraint = (IntVar != 0);

    fscanf(f,"scaleFact %f\n",&scaleFactor);

    fscanf(f,"fixedChartClusters %d\n",&fixedChartClusters);
    fscanf(f,"useFlowSolver %d\n",&IntVar);
    parameters.useFlowSolver = IntVar;
    std::cout << "useFlowSolver: " << parameters.useFlowSolver << std::endl;

    std::array<char, 1024> filename = {0};

    int ret = fscanf(f,"flow_config_filename \"%1000[^\"]\"\n",filename.data());
    parameters.flow_config_filename = filename.data();
    std::cout << "flow_config_filename: " << parameters.flow_config_filename << std::endl;

    std::fill(filename.begin(), filename.end(), 0);

    ret = fscanf(f,"satsuma_config_filename \"%1000[^\"]\"\n",filename.data());
    parameters.satsuma_config_filename = filename.data();
    std::cout << "satsuma_config_filename: " << parameters.satsuma_config_filename << std::endl;
    (void)ret;
    fclose(f);
}

void SaveSetupFile(const std::string& path, QuadRetopology::Parameters& parameters, float& scaleFactor, int& fixedChartClusters)
{
    FILE *f=fopen(path.c_str(),"wt");
    assert(f!=NULL);

    fprintf(f,"alpha %f\n", parameters.alpha);
    fprintf(f,"ilpMethod %d\n", parameters.ilpMethod==QuadRetopology::ILPMethod::ABS ? 0 : 1);
    fprintf(f,"timeLimit %f\n", parameters.timeLimit);
    fprintf(f,"gapLimit %f\n", parameters.gapLimit);

    fprintf(f,"callbackTimeLimit %d", static_cast<int>(parameters.callbackTimeLimit.size()));
    for (float& time : parameters.callbackTimeLimit) fprintf(f," %f", time);
    fprintf(f,"\n");

    fprintf(f,"callbackGapLimit %d", static_cast<int>(parameters.callbackGapLimit.size()));
    for (float& gap : parameters.callbackGapLimit) fprintf(f," %f", gap);
    fprintf(f,"\n");

    fprintf(f,"minimumGap %f\n", parameters.minimumGap);
    fprintf(f,"isometry %d\n", parameters.isometry ? 1 : 0);
    fprintf(f,"regularityQuadrilaterals %d\n", parameters.regularityQuadrilaterals ? 1 : 0);
    fprintf(f,"regularityNonQuadrilaterals %d\n", parameters.regularityNonQuadrilaterals ? 1 : 0);
    fprintf(f,"regularityNonQuadrilateralsWeight %f\n", parameters.regularityNonQuadrilateralsWeight);
    fprintf(f,"alignSingularities %d\n", parameters.alignSingularities ? 1 : 0);
    fprintf(f,"alignSingularitiesWeight %f\n", parameters.alignSingularitiesWeight);
    fprintf(f,"repeatLosingConstraintsIterations %d\n", parameters.repeatLosingConstraintsIterations);
    fprintf(f,"repeatLosingConstraintsQuads %d\n", parameters.repeatLosingConstraintsQuads ? 1 : 0);
    fprintf(f,"repeatLosingConstraintsNonQuads %d\n", parameters.repeatLosingConstraintsNonQuads ? 1 : 0);
    fprintf(f,"repeatLosingConstraintsAlign %d\n", parameters.repeatLosingConstraintsAlign ? 1 : 0);
    fprintf(f,"hardParityConstraint %d\n", parameters.hardParityConstraint ? 1 : 0);
    fprintf(f,"scaleFact %f\n", static_cast<float>(scaleFactor));
    fprintf(f,"fixedChartClusters %d\n", fixedChartClusters);

    fclose(f);
}

int main(int argc, char* argv[])
{
    try {
        return actual_main(argc, argv);
    }
    catch (std::runtime_error& e) {
        std::cerr << "fatal error: " << e.what() << std::endl;
        return 1;
    }
}

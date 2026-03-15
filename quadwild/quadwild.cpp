//***************************************************************************/
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

#include <iomanip>
#include <clocale>
#include <cstring>

#include "functions.h"
#include <qw_checkpoint.h>

#ifdef _WIN32
#  include <windows.h>
#  include <stdlib.h>
#  include <errhandlingapi.h>
#endif

// ============================================================
// Checkpoint flags for the 3-step quadwild pipeline
// ============================================================

struct QuadwildCheckpointFlags {
    std::string save_dir;
    bool save_all = false;
    int run_from_step = 1;  // 1=remesh, 2=trace, 3=quadrangulate
    int run_to_step = 3;
    bool list_stages = false;
};

static void parse_flags(int argc, char* argv[], QuadwildCheckpointFlags& f) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-save-dir") == 0 && i + 1 < argc) {
            f.save_dir = argv[++i];
        } else if (strcmp(argv[i], "-save-all") == 0) {
            f.save_all = true;
        } else if (strcmp(argv[i], "-run-from") == 0 && i + 1 < argc) {
            f.run_from_step = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-run-to") == 0 && i + 1 < argc) {
            f.run_to_step = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-list-stages") == 0) {
            f.list_stages = true;
        }
    }
}

int actual_main(int argc, char* argv[])
{
#ifdef _WIN32
    SetErrorMode(0);
#endif
    Parameters parameters;

    FieldTriMesh trimesh;
    TraceMesh traceTrimesh;

    TriangleMesh trimeshToQuadrangulate;
    std::vector<std::vector<size_t>> trimeshPartitions;
    std::vector<std::vector<size_t>> trimeshCorners;
    std::vector<std::pair<size_t,size_t> > trimeshFeatures;
    std::vector<size_t> trimeshFeaturesC;

    PolyMesh quadmesh;
    std::vector<std::vector<size_t>> quadmeshPartitions;
    std::vector<std::vector<size_t>> quadmeshCorners;
    std::vector<int> ilpResult;

    // Parse checkpoint flags
    QuadwildCheckpointFlags ckpt;
    parse_flags(argc, argv, ckpt);

    if (ckpt.list_stages) {
        printf("QuadWild pipeline stages:\n");
        printf("  1: Remesh and field computation\n");
        printf("  2: Field tracing\n");
        printf("  3: Quadrangulation (uses quad_from_patches stages internally)\n");
        printf("\nFor fine-grained checkpointing within step 3, use quad_from_patches directly.\n");
        printf("Use -run-from N -run-to M to run steps N through M.\n");
        return 0;
    }

    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0]
                  << " <mesh.{obj,ply}>"
                     " [1|2|3]"
                     " [*.sharp|*.rosy|*.txt]....\n"
                     " The 1|2|3 parameter determines after which step to stop:\n"
                     "   1: Remesh and field\n"
                     "   2: Tracing\n"
                     "   3: Quadrangulation (default)\n"
                     "\n"
                     " Checkpoint flags:\n"
                     "   -save-dir DIR      Directory for intermediate files\n"
                     "   -save-all          Save after every step\n"
                     "   -run-from N        Start from step N (1-3)\n"
                     "   -run-to N          Stop after step N (1-3)\n"
                     "   -list-stages       List all stages\n";
        return 1;
    }

    std::string meshFilename;
    int stopAfterStep = 3;

    // Parse positional args (skip checkpoint flags)
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-save-dir") == 0 ||
            strcmp(argv[i], "-run-from") == 0 ||
            strcmp(argv[i], "-run-to") == 0) {
            ++i;
            continue;
        }
        if (strcmp(argv[i], "-save-all") == 0 ||
            strcmp(argv[i], "-list-stages") == 0) {
            continue;
        }
        pos_args.push_back(argv[i]);
    }

    meshFilename = pos_args[0];
    if (meshFilename.size() < 4) {
        std::cerr << "invalid mesh filename '" << meshFilename << "'" << std::endl;
        return 1;
    }
    auto meshFilenamePrefix = std::string(meshFilename.begin(), meshFilename.end() - 4);

    if (pos_args.size() >= 2) {
        int s = std::atoi(pos_args[1].c_str());
        if (s >= 1 && s <= 3) stopAfterStep = s;
    }

    // Apply checkpoint overrides
    if (ckpt.run_to_step < stopAfterStep) stopAfterStep = ckpt.run_to_step;
    int startStep = ckpt.run_from_step;

    std::string sharpFilename;
    std::string fieldFilename;

    std::setlocale(LC_NUMERIC, "en_US.UTF-8");

    std::cout<<"Reading input..."<<std::endl;

    std::string configFile = "basic_setup.txt";
    for (size_t i = 2; i < pos_args.size(); ++i) {
        std::string pathTest = pos_args[i];

        if (pathTest.find(".sharp") != std::string::npos) {
            sharpFilename = pathTest;
            parameters.hasFeature = true;
            continue;
        }
        if (pathTest.find(".txt") != std::string::npos) {
            configFile = pathTest;
            loadConfigFile(pathTest.c_str(), parameters);
            continue;
        }
        if (pathTest.find(".rosy") != std::string::npos) {
            fieldFilename = pathTest;
            parameters.hasField = true;
            continue;
        }
    }
    loadConfigFile(configFile, parameters);

    std::cout<<"Loading:"<<meshFilename.c_str()<<std::endl;

    bool allQuad;
    bool loaded=trimesh.LoadTriMesh(meshFilename,allQuad);
    trimesh.UpdateDataStructures();

    if (!loaded) {
        std::cout<<"Wrong mesh filename"<<std::endl;
        exit(0);
    }

    std::cout<<"Loaded "<<trimesh.fn<<" faces and "<<trimesh.vn<<" vertices"<<std::endl;

    // ============================================================
    // Step 1: Remesh and field
    // ============================================================

    if (startStep <= 1) {
        std::cout<<std::endl<<"--------------------- 1 - Remesh and field ---------------------"<<std::endl;
        remeshAndField(trimesh, parameters, meshFilename, sharpFilename, fieldFilename);
        std::cout << "[CHECKPOINT] Step 1 complete. Output files written to disk." << std::endl;
    } else {
        std::cout << "[CHECKPOINT] Skipping step 1 (run-from=" << startStep << ")" << std::endl;
    }
    if (stopAfterStep == 1) return 0;

    // ============================================================
    // Step 2: Tracing
    // ============================================================

    if (startStep <= 2) {
        std::cout<<std::endl<<"--------------------- 2 - Tracing ---------------------"<<std::endl;
        meshFilenamePrefix += "_rem";
        trace(meshFilenamePrefix, traceTrimesh);
        std::cout << "[CHECKPOINT] Step 2 complete. Output files written to disk." << std::endl;
    } else {
        std::cout << "[CHECKPOINT] Skipping step 2 (run-from=" << startStep << ")" << std::endl;
        meshFilenamePrefix += "_rem";
    }
    if (stopAfterStep == 2) return 0;

    // ============================================================
    // Step 3: Quadrangulation
    // ============================================================

    if (startStep <= 3) {
        std::cout<<std::endl<<"--------------------- 3 - Quadrangulation ---------------------"<<std::endl;
        quadrangulate(meshFilenamePrefix + ".obj", trimeshToQuadrangulate, quadmesh,
            trimeshPartitions, trimeshCorners, trimeshFeatures, trimeshFeaturesC,
            quadmeshPartitions, quadmeshCorners, ilpResult, parameters);
        std::cout << "[CHECKPOINT] Step 3 complete." << std::endl;
    }

    return 0;
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

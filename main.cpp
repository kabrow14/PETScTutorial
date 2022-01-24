#include <iostream>

#include <petscTutorialTestFunctions.h>

int main(int argc, char **argv) {
    // return TutorialFun::HelloWorld(argc, argv);
    // return TutorialFun::VectorBuilding(argc, argv);
    // return TutorialFun::MatrixBuilding(argc, argv);
    return TutorialFun::SolveTridiagKSP(argc, argv);
    // return TutorialFun::AdvancedMatrixAssembly(argc, argv);
    return 0;
}

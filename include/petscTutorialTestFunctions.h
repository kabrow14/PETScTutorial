#ifndef PETSCTUTORIAL_PETSCTUTORIALTESTFUNCTIONS_H
#define PETSCTUTORIAL_PETSCTUTORIALTESTFUNCTIONS_H

#include <petsc.h>

namespace TutorialFun {

    PetscErrorCode HelloWorld(int argc, char **argv) {
        // Error code and rank
        PetscErrorCode ierr;
        PetscMPIInt rank;

        // Initialize
        PetscInitialize(&argc, &argv, nullptr, nullptr); if (ierr) return ierr;

        // Create the ranks (processors) for PETSC
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        CHKERRQ(ierr);

        // Hello world for PETSc in parallel
        ierr = PetscPrintf(PETSC_COMM_SELF, "From proc %d: Hello, World!\n", rank);
        CHKERRQ(ierr);

        /** IF USING PETSC_COMM_WORLD to get process synchronized output:
         * ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "MESSAGE", rank); CHKERRQ(ierr);
         * ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD); CHKERRQ(ierr);
         */


        // Finalize
        return PetscFinalize();
    }

    PetscErrorCode VectorBuilding(int argc, char **argv) {

        const char help[] = "VectorBuilding Function using PETSc \n\n";
        // variables needed
        PetscErrorCode ierr;
        PetscMPIInt    rank;
        PetscInt       i, istart, iend, N=10; // i is declared above apparently this is the way things are done
        PetscReal      norm;
        PetscScalar    v, one = 1.0, two = 2.0, dot;
        Vec            x, y;


        // Initialize
        PetscInitialize(&argc, &argv, nullptr, help); if (ierr) return ierr;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        // Vector encapsulates PetscScalar array
        /** Vector lifecycle
         *
         * Variables
         * Vec v; PetscInt m=2, M=8;
         * VecType type=VECMPI;             // FOR SEQUENTIAL -> type=VECSEQ
         * MPI_Comm comm=PETSC_COMM_WORLD;  // FOR SEQUENTIAL -> comm=PETSC_COMM_SELF
         *
         *  Create   : VecCreate(comm, &v); //
         *  Set Size : VecSetSizes(v, m, M); // FOR SEQUENTIAL -> VecSetSizes(v, M, M);
         *  Set Type : VecSetType(v, type);
         *  Set Opts : VecSetFromOptions(v); -> control property of vector from command line
         *  Destroy  : VecDestroy(&v);
         *
         *  All in one:
         *  VecCreateMPI(comm, m, M, &v); // VecCreateSeq(comm, M, &v);
         */

        // Parallel layout
        // Consider vector v with local size m, global size M, distributed across 3 processes
        // Call VecSetSizes(v, m, M) -> Note be careful this could get difficult to manage

        // set either m or M to PETSC_DECIDE lets PETSc use the standard layout;
        // get this standard layout across comm: PetscSplitOwnership(comm, &m, &M)

        // Querying layout is simple enough with local and global sizes:
        // VecGetLocalSize(v, &m) and VecGetSize(v, &M)
        // global indices of the first and last elements of the local portion:
        // VecGetOwnershipRange(v, &lo, &hi)

        // HERE'S HOW TO ASSEMBLE THE VECTORS USING MPI:
        // ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x); CHKERRQ(ierr);
        // TO DO USING VecCreate when mpi is not known
        ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
        ierr = VecSetSizes(x, PETSC_DECIDE, N); CHKERRQ(ierr); // petsc will decide global size later
        ierr = VecSetFromOptions(x); CHKERRQ(ierr);
        ierr = VecSetUp(x); CHKERRQ(ierr);
        ierr = VecDuplicate(x, &y); CHKERRQ(ierr); // Create duplicate of x using y


        ierr = VecSet(x, one); CHKERRQ(ierr);      // set all values in vector to 1.0
        ierr = VecSet(y, two); CHKERRQ(ierr);      // set all values in vector to 1.0
        // ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr); // std::cout
        // ierr = PetscPrintf(PETSC_COMM_WORLD, "-------------\n"); CHKERRQ(ierr);

        VecGetOwnershipRange(x,&istart,&iend); // Get the bounds of the vector

        for (i=istart; i<iend; i++) {
            v = static_cast<PetscScalar>(i);
            ierr = VecSetValues(x,1,&i,&v,ADD_VALUES); CHKERRQ(ierr);
        }

        ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

        ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(y); CHKERRQ(ierr);


        ierr = VecScale(x, two); CHKERRQ(ierr);
        ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

        ierr = VecDot(x, y, &dot); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Dot of x and y: %g\n",dot); CHKERRQ(ierr);
        ierr = VecNorm(x,NORM_2,&norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of x: %g\n",norm); CHKERRQ(ierr);

        // ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        ierr = VecDestroy(&x); CHKERRQ(ierr);

        // Finalize
        return PetscFinalize();
    }

    PetscErrorCode MatrixBuilding(int argc, char **argv) {
        const char help[] = "MatrixBuilding Function using PETSc \n\n";
        // variables needed
        Mat            A;
        Vec            x, b;
        PetscErrorCode ierr;
        PetscMPIInt    rank;
        PetscInt       i, N=10, col[3], rstart, rend, nlocal, mlocal;
        PetscScalar    value[3];
        ierr = PetscInitialize(&argc, &argv, nullptr, help); if (ierr) return ierr;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        // Build matrix
        ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
        ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N); CHKERRQ(ierr); // Let PETSC decide how to partition across procs
        ierr = MatSetFromOptions(A); CHKERRQ(ierr); // Check for command line arguments
        ierr = MatSetUp(A); CHKERRQ(ierr);

        // Here we'll set values in the matrix
        // A processor can set any entry in a matrix even if entry is not locally owned, but efficient assembly is important.
        // Hence, we identify the portion of the matrix each processor contains and insert values into the matrix.
        // Insert values one row at a time in a loop by providing the column indices and values.
        // Handle the boundaries separately since these rows only contain two entries.
        // HERE -> SPARSE TRI-DIAGONAL MATRIX IS MADE
        ierr = MatGetOwnershipRange(A, &rstart, &rend); CHKERRQ(ierr);
        /** Boundary: whichever rank has boundary elements -> handle them accordingly */
        if (!rstart) {
            rstart = 1;
            i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
            ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (rend == N) {
            rend = N-1;
            i    = N-1; col[0] = N-2; col[1] = N-1; value[0] = -1.0; value[1] = 2.0;
            ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }

        /** Interior */
        value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
        for (i=rstart; i<rend; i++) {
            col[0] = i-1; col[1] = i; col[2] = i+1;
            ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }

        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


        // Below, we'll create a vector compatible with matrix A
        ierr = MatGetLocalSize(A,&nlocal,&mlocal);CHKERRQ(ierr); // get rank local row and col indices
        // Create vector
        ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
        ierr = VecSetSizes(x,mlocal,N);CHKERRQ(ierr);
        ierr = VecSetFromOptions(x);CHKERRQ(ierr);
        ierr = VecDuplicate(x,&b);CHKERRQ(ierr);   // duplicate for vector b
        ierr = VecSet(x,1.0);CHKERRQ(ierr);        // initialize vector x to ones

        // Assemble vectors
        ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

        ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b); CHKERRQ(ierr);


        // Matrix multiplication Ax = b
        ierr = MatMult(A,x,b);CHKERRQ(ierr);
        ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

        // Scale A by 2.0 A = 2 * I * A
        ierr = MatScale(A,2.0);CHKERRQ(ierr);
        ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

        ierr = MatDestroy(&A);CHKERRQ(ierr);
        ierr = VecDestroy(&x);CHKERRQ(ierr);
        ierr = VecDestroy(&b);CHKERRQ(ierr);

        return PetscFinalize();
    }

    PetscErrorCode SolveTridiagKSP(int argc, char **argv) {

        const char help[] = "Solve Tridiagonal System using KSP\n\n";

        Vec            x, b, u;      /* approx solution, RHS, exact solution */
        Mat            A;            /* linear system matrix */
        KSP            ksp;          /* linear solver context */
        PC             pc;           /* preconditioner context */
        PetscReal      norm;         /* norm of solution error */
        PetscErrorCode ierr;
        PetscInt       i,n = 10, col[3], its, rstart, rend, nlocal, mlocal;
        PetscMPIInt    rank;
        PetscScalar    value[3];

        PetscInitialize(&argc, &argv, nullptr, help); if (ierr) return ierr;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        /*
        Create matrix.  When using MatCreate(), the matrix format can
        be specified at runtime.

        Performance tuning note:  For problems of substantial size,
        preallocation of matrix memory is crucial for attaining good
        performance. See the matrix chapter of the users manual for details.
        */
        ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
        ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
        ierr = MatSetFromOptions(A);CHKERRQ(ierr);
        ierr = MatSetUp(A);CHKERRQ(ierr);

        /*
        Assemble matrix
        */
        ierr = MatGetOwnershipRange(A, &rstart, &rend); CHKERRQ(ierr);
        /** Boundary: whichever rank has boundary elements -> handle them accordingly */
        if (!rstart) {
            rstart = 1;
            i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
            ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }
        if (rend == n) {
            rend = n-1;
            i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
            ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }

        /** Interior */
        value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
        for (i=rstart; i<rend; i++) {
            col[0] = i-1; col[1] = i; col[2] = i+1;
            ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        }

        // A is symmetric -> enable option for icc preconditioner
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);

        /*
        Create vectors.  Note that we form 1 vector from scratch and
        then duplicate as needed.
        */
        // Below, we'll create a vector compatible with matrix A
        ierr = MatGetLocalSize(A,&nlocal,&mlocal);CHKERRQ(ierr); // get rank local row and col indices
        // Create vector
        ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
        ierr = VecSetSizes(x,mlocal,n);CHKERRQ(ierr);
        ierr = VecSetFromOptions(x);CHKERRQ(ierr);
        ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
        ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

        /*
        Set exact solution; then compute right-hand-side vector.
        */
        ierr = VecSet(u,1.0);CHKERRQ(ierr);
        ierr = MatMult(A,u,b);CHKERRQ(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create the linear solver and set various options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

    /*
    Set operators. Here the matrix that defines the linear system
    also serves as the matrix that defines the preconditioner.
    */
        ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    /*
    Set linear solver defaults for this problem (optional).
   - By extracting the KSP and PC contexts from the KSP context,
     we can then directly call any KSP and PC routines to set
     various options.
   - The following four statements are optional; all of these
     parameters could alternatively be specified at runtime via
     KSPSetFromOptions();
    */
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

/*
  Set runtime options, e.g.,
      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  These options will override those specified above as long as
  KSPSetFromOptions() is called _after_ any other customization
  routines.
*/
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);


        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Solve the linear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

/*
   View solver info; we could instead use the option -ksp_view to
   print this info to the screen at the conclusion of KSPSolve().
*/
        ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Check the solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
        ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                           "Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

        /*
Free work space.  All PETSc objects should be destroyed when they
are no longer needed.
*/
        ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
        ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
        ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

        /*
           Always call PetscFinalize() before exiting a program.  This routine
             - finalizes the PETSc libraries as well as MPI
             - provides summary and diagnostic information if certain runtime
               options are chosen (e.g., -log_view).
        */


        return PetscFinalize();
    }

    PetscErrorCode AdvancedMatrixAssembly(int argc, char **argv) {

        Mat            A;
        PetscErrorCode ierr;
        PetscMPIInt    rank;
        PetscInt       m[3], n, M, N;


        static char help[] = "Advanced Matrix Assembly tutorial test function\n\n";
        ierr = PetscInitialize(&argc, &argv, nullptr, help); if (ierr) return ierr;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        // We'll create the following matrix
        /**
         *          1  2  0  |  0  3  0  |  0  4
           Proc 0   0  5  6  |  7  0  0  |  8  0
                    9  0 10  | 11  0  0  | 12  0
           -------------------------------------
                   13  0 14  | 15 16 17  |  0  0
           Proc1    0 18  0  | 19 20 21  |  0  0
                    0  0  0  | 22 23  0  | 24  0
           -------------------------------------
           Proc2  25 26 27  |  0  0 28  | 29  0
                  30  0  0  | 31 32 33  |  0 34
         */

        /** This corresponds to the follwoing submatrices
         *             [ A B C ]
             Ablock =  [ D E F ]
                       [ G H I ]
         */

         /* Allocating local number of rows for each process:
          * -----------------------------------------------------------------------
          * We can see that 'm' for proc0, proc1, and proc3
          * are 3, 3, 2 respectively where m is the number of
          * local rows.
          */
          m[0] = 3; m[1] = 3; m[2] = 2;
          /* M and N are 8 since the matrix is 8x8 */



        return PetscFinalize();
    }

    /**
    PetscErrorCode BlockMatrixAssembly() {

        return PetscFinalize();
    }
    */

} // TutorialFun

#endif //PETSCTUTORIAL_PETSCTUTORIALTESTFUNCTIONS_H

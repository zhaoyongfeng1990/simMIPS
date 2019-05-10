#include "particles.h"

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    clock_t it, ft;
    it = clock();
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    init_genrand64(pid + 100);

    // OLDTYPES[0]=MPI_DOUBLE;
    // OLDTYPES[1]=MPI_UINT64_T;

    //Define the MPI datatype for particles
    int blocklengths = 5;
#ifdef QuorumSensing
    ++blocklengths;
#endif
#ifdef RTP
    ++blocklengths;
#endif
    MPI_Type_contiguous(blocklengths, MPI_DOUBLE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);

    probePlate probe_l, probe_r;          //Left and right probe
    uint64_t NParticlesLocal[NProcesses]; //Number of particles in the local process
    box *bList = NULL;                    //Array of boxes
    particle *pList = NULL;               //Used for collecting data from all other processes

    box *UpperBoundary = (box *)calloc(XLocalSize + 2, sizeof(box));
    box *LowerBoundary = (box *)calloc(XLocalSize + 2, sizeof(box));
    box *LeftBoundary = (box *)calloc(YLocalSize, sizeof(box));
    box *RightBoundary = (box *)calloc(YLocalSize, sizeof(box));

    box *FirstLine = (box *)calloc(XLocalSize + 2, sizeof(box)); //Also used to store the particles to exchange
    box *LastLine = (box *)calloc(XLocalSize + 2, sizeof(box));
    box *FirstCol = (box *)calloc(YLocalSize, sizeof(box)); //Also used to store the particles to exchange
    box *LastCol = (box *)calloc(YLocalSize, sizeof(box));
    initialization(&bList, &probe_l, &probe_r);
    initialBoxB(UpperBoundary, XLocalSize + 2);
    initialBoxB(LowerBoundary, XLocalSize + 2);
    initialBoxB(FirstLine, XLocalSize + 2);
    initialBoxB(LastLine, XLocalSize + 2);
    initialBoxB(LeftBoundary, YLocalSize);
    initialBoxB(RightBoundary, YLocalSize);
    initialBoxB(FirstCol, YLocalSize);
    initialBoxB(LastCol, YLocalSize);

#ifndef CircularInitialCondition
    //Load(bList, &probe_l, &probe_r, &NParticlesLocal[pid]);
    //probe.k=0.01;
    putParticles(bList, &probe_l, &probe_r, &NParticlesLocal[pid]);
#else
    putParticlesCircular(bList, &NParticlesLocal[pid]);
#endif
    MPI_Allreduce(&NParticlesLocal[pid], &NParticles, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    printf("%d: %lld %lld\n", pid, NParticlesLocal[pid], NParticles);
    if (pid == 0)
    {
        //Used for collecting data from all other processes
        pList = (particle *)malloc(NParticles * sizeof(particle));
    }

    updateBoundaryInformation(bList, UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary);
    updateForces(bList, 0, UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary, &probe_l, &probe_r);

#ifdef QuorumSensing
    bufferSmall=(double*)malloc(NParticles*sizeof(double));
    bufferHuge_p=(uint32_t*)malloc(NParticles*sizeof(uint32_t));
    updateVelocities(bList, 0, NParticlesLocal[pid], UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary);
#endif

    int timepoint = 0;
    collectingResult(bList, pList);
    FILE *outfile = NULL;
    FILE *outfileObj = NULL;
    if (pid == 0)
    {
        char filename[11]; //File name will be like p0000.txt
        char buffer[6];
        sprintf(buffer, "%05d", timepoint);
        strcpy(filename, "p");
        strcat(filename, buffer);
        strcat(filename, ".txt");
        outfile = fopen(filename, "w");
        outfileObj = fopen("object.txt", "w");
        fprintf(outfileObj, "%e %e %e %e %e ", probe_l.x, probe_l.y, probe_l.fx, probe_l.fy, probe_l.k);
        fprintf(outfileObj, "%e %e %e %e %e\n", probe_r.x, probe_r.y, probe_r.fx, probe_r.fy, probe_r.k);
        for (uint64_t i = 0; i < NParticles; i++)
        {
            fprintf(outfile, "%e %e %e ", pList[i].x, pList[i].y, pList[i].theta);
        }
        fprintf(outfile, "\n");
        fflush(outfile);
        fflush(outfileObj);
        fclose(outfile);
        ++timepoint;
    }
    for (uint64_t steps = 0; steps < TotalSteps; steps++)
    {
        step(bList, steps, &NParticlesLocal[pid], FirstLine, LastLine, FirstCol, LastCol, &probe_l, &probe_r);
        exchangeParticles(bList, &NParticlesLocal[pid], UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary, FirstLine, LastLine, FirstCol, LastCol);
        updateBoundaryInformation(bList, UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary);

        updateForces(bList, steps, UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary, &probe_l, &probe_r);
#ifdef QuorumSensing
        updateVelocities(bList, steps, NParticlesLocal[pid], UpperBoundary, LowerBoundary, LeftBoundary, RightBoundary);
#endif

        if (steps % (1 * StepsPerSec) == 1 * StepsPerSec - 1)
        {
            Save(bList, &probe_l, &probe_r, NParticlesLocal[pid]);
            if (pid == 0)
            {
                fprintf(outfileObj, "%e %e %e %e %e ", probe_l.x, probe_l.y, probe_l.fx, probe_l.fy, probe_l.k);
                fprintf(outfileObj, "%e %e %e %e %e\n", probe_r.x, probe_r.y, probe_r.fx, probe_r.fy, probe_r.k);
                fflush(outfileObj);
            }
        }

        if (steps % (100 * StepsPerSec) == 100 * StepsPerSec - 1)
        {
            collectingResult(bList, pList);
            if (pid == 0)
            {
                char filename[11]; //file name will be like p0000.txt
                char buffer[6];
                sprintf(buffer, "%05d", timepoint);
                strcpy(filename, "p");
                strcat(filename, buffer);
                strcat(filename, ".txt");
                outfile = fopen(filename, "w");
                for (uint64_t i = 0; i < NParticles; i++)
                {
                    fprintf(outfile, "%e %e %e ", pList[i].x, pList[i].y, pList[i].theta);
                }
                fprintf(outfile, "\n");
                fflush(outfile);
                fclose(outfile);
                ++timepoint;
            }
        }
    }

    if (pid == 0)
    {
        free(pList);
        fclose(outfileObj);
    }

    destruction(bList);
    for (int i = 0; i < XLocalSize + 2; ++i)
    {
        free(UpperBoundary[i].partList);
        free(LowerBoundary[i].partList);
        free(FirstLine[i].partList);
        free(LastLine[i].partList);
    }
    for (int i = 0; i < YLocalSize; ++i)
    {
        free(LeftBoundary[i].partList);
        free(RightBoundary[i].partList);
        free(FirstCol[i].partList);
        free(LastCol[i].partList);
    }
    free(UpperBoundary);
    free(LowerBoundary);
    free(FirstLine);
    free(LastLine);
    free(LeftBoundary);
    free(RightBoundary);
    free(FirstCol);
    free(LastCol);
    
#ifdef QuorumSensing
    free(bufferSmall);
    free(bufferHuge_p);
    #endif

    MPI_Type_free(&MPI_PARTICLE);
    ft = clock();
    printf("%f\n", (double)(ft - it) / CLOCKS_PER_SEC);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

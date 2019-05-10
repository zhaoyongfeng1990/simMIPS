#ifndef PARTICLES_H
#define PARTICLES_H

#include <cstddef>
#include <cstdbool>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <cstring>
#include "config.h"
#include "mt64.h"

//A library which enables SIMD instructions
//Will speed up if using SSE or AVX instruction sets
#define VCL_NAMESPACE vcl
#include "./vectorclass/vectormath_exp.h"
#include "./vectorclass/vectormath_trig.h"
#include "./vectorclass/vectormath_hyp.h"

using namespace std;
using namespace vcl;

typedef struct
{
    double x;
    double y;
    double theta;
    double fx;
    double fy;
#ifdef RTP
    double nextTumbTime;
#endif
#ifdef QuorumSensing
    double v;
#endif
} particle;

typedef struct
{
    particle *partList;
    uint64_t NumParticles;
    uint64_t Capacity;
} box;

typedef struct
{
    double x;
    double y;
    double vx;
    double vy;
    double fx;
    double fy;
    double k;
} probePlate;

#ifdef PairwiseForces
const double v = V;
#endif

static box EmptyBox = {NULL, 0, 0};
static MPI_Datatype MPI_PARTICLE;

#ifdef QuorumSensing
//Buffers for calculating the local densities
double *bufferSmall = NULL;    //Serializated buffur for calculated speed
uint32_t *bufferHuge_p = NULL; //Serializated buffer for particle number
#endif

void deleteParticleInBox(box *__restrict__ bList, const uint64_t bidx, const uint64_t pidx)
{
    --bList[bidx].NumParticles; //the particle number in the old site reduced by 1
    bList[bidx].partList[pidx] = bList[bidx].partList[bList[bidx].NumParticles];
}

void addParticleInBox(box *__restrict__ bList, const particle &__restrict__ p, const uint64_t bidx)
{
    if (bList[bidx].NumParticles == bList[bidx].Capacity)
    {
        particle *temp = (particle *)realloc((void *)bList[bidx].partList, bList[bidx].Capacity * 2 * sizeof(particle));
        if (temp != NULL)
            bList[bidx].partList = temp;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        bList[bidx].Capacity *= 2;
    }
    bList[bidx].partList[bList[bidx].NumParticles] = p;
    ++bList[bidx].NumParticles;
}

void moveParticleInBox(box *__restrict__ bList, uint64_t pidx, uint64_t bidx, uint64_t bidx_dest)
{
    particle temp = bList[bidx].partList[pidx];
    deleteParticleInBox(bList, bidx, pidx);
    addParticleInBox(bList, temp, bidx_dest);
}

void sampleDirection(double *__restrict__ ux, double *__restrict__ uy)
{
    double r = genrand64_real2() * (2.0 * PI);
    *ux = cos(r);
    *uy = sqrt(1.0 - (*ux) * (*ux)) * (2 * (r < PI) - 1);
}

#ifdef PairwiseForces
void forceFunction(double rx, double ry, double *__restrict__ fx, double *__restrict__ fy)
{
    double r2 = rx * rx + ry * ry;

    if (r2 < BoxLength2)
    {
#ifdef WCAForces
        double r4 = r2 * r2;
        double r8 = r4 * r4;
        double r6 = r2 * r4;
        double rdVdr = 24.0 * InteractionEpsilon * Sigma6 * (2.0 * Sigma6 / r6 - 1.0) / r8;
        *fx = rx * rdVdr;
        *fy = ry * rdVdr;
#endif
#ifdef HarmonicForces
        double r = sqrt(r2);
        *fx = (InteractionEpsilon / r - (InteractionEpsilon / Sigma)) * rx;
        *fy = (InteractionEpsilon / r - (InteractionEpsilon / Sigma)) * ry;
#endif
    }
    else
    {
        *fx = 0;
        *fy = 0;
    }
}
#endif

#ifdef QuorumSensing

//Calculating the local density with an exponential kernal
// void densityKernalArray(double *buffer, const uint64_t size)
// {
//     Vec4d temp, r;
//     int64_t i;
//     for (i = 0; i < (int64_t)size - 4; i += 4)
//     {
//         temp.load(buffer + i);
//         r = exp(temp);
//         r *= Zinv;
//         r.store(buffer + i);
//     }
//     temp.load_partial(size - i, buffer + i);
//     r = exp(temp);
//     r *= Zinv;
//     r.store_partial(size - i, buffer + i);
//     //return (r2 > R0 * R0) * exp(-1.0 / (1.0 - r2 / (R0 * R0))) * Zinv;
// }

//Set the velocity of QSAPs using the SIMD instructions
void setQSVelocityArray(double *__restrict__ buffer, const uint64_t size)
{
    Vec4d temp, r;
    int64_t i;
    for (i = 0; i < (int64_t)size - 4; i += 4)
    {
        temp.load(buffer + i);
        temp -= Rhom;
        temp /= Lf;
        r = tanh(temp);
        r += 1.0;
        r *= 0.5 * (V1 - V0);
        r += V0;
        r.store(buffer + i);
    }
    temp.load_partial(size - i, buffer + i);
    temp -= Rhom;
    temp /= Lf;
    r = tanh(temp);
    r += 1.0;
    r *= 0.5 * (V1 - V0);
    r += V0;
    r.store_partial(size - i, buffer + i);
}
#endif

inline double wallForce(double pos)
{
    //return - 4.0 * WallK * pos * pos * pos;
    double p2 = pos * pos;
    double p4 = p2 * p2;
    double p6 = p2 * p4;
    double p7 = p6 * pos;
    const double s = BoxLength * XWall;
    const double s2 = s * s;
    const double s3 = s * s2;
    const double s6 = s3 * s3 * 0.5;
    return 24.0 * WallK * s6 * (2.0 * s6 / p6 - 1.0) / p7;
}

void updateForces(box *__restrict__ bList, const uint64_t steps,
                  box *__restrict__ UpperBoundary, box *__restrict__ LowerBoundary,
                  box *__restrict__ LeftBoundary, box *__restrict__ RightBoundary,
                  probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

#ifdef PairwiseForces
    for (int32_t by = 0; by < YLocalSize; ++by)
    {
        for (int32_t bx = 0; bx < XLocalSize; ++bx)
        {
            uint64_t bidx = by * XLocalSize + bx;

            box *pbox[8]; //List of boxes to search
            for (int i = 0; i < 8; ++i)
                pbox[i] = &EmptyBox;

            //Neighbour boxes are labeled as
            // 0 1 2
            // 3   4
            // 5 6 7
            //Notice: in the words
            //the y-axis is taken to be downwards
            //the x-axis is taken to be rightwards

            if (by == 0)
            {
#ifdef WithYWalls
                if (pyid != 0)
#endif
                {
                    pbox[0] = UpperBoundary + bx;
                    pbox[1] = UpperBoundary + bx + 1;
                    pbox[2] = UpperBoundary + bx + 2;
                }
            }

            if (bx == 0)
            {
#ifdef WithXWalls
                if (pxid != 0)
#endif
                {
                    if (by != 0)
                        pbox[0] = LeftBoundary + by - 1;
                    pbox[3] = LeftBoundary + by;
                    if (by != YLocalSize - 1)
                        pbox[5] = LeftBoundary + by + 1;
                }
            }

            if (bx == XLocalSize - 1)
            {
#ifdef WithXWalls
                if (pxid != NPx - 1)
#endif
                {
                    if (by != 0)
                        pbox[2] = RightBoundary + by - 1;
                    pbox[4] = RightBoundary + by;
                    if (by != YLocalSize - 1)
                        pbox[7] = RightBoundary + by + 1;
                }
            }
            else
            {
                pbox[4] = bList + bx + 1 + by * XLocalSize;
            }

            if (by == YLocalSize - 1)
            {
#ifdef WithYWalls
                if (pyid != NPy - 1)
#endif
                {
                    pbox[5] = LowerBoundary + bx;
                    pbox[6] = LowerBoundary + bx + 1;
                    pbox[7] = LowerBoundary + bx + 2;
                }
            }
            else
            {
                if (bx != 0)
                    pbox[5] = bList + bx + (by + 1) * XLocalSize - 1;
                pbox[6] = bList + bx + (by + 1) * XLocalSize;
                if (bx != XLocalSize - 1)
                    pbox[7] = bList + bx + (by + 1) * XLocalSize + 1;
            }

            for (uint64_t ip1 = 0; ip1 < bList[bidx].NumParticles; ip1++)
            {
                for (uint64_t ip2 = ip1 + 1; ip2 < bList[bidx].NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - bList[bidx].partList[ip2].x;
                    const double ry = bList[bidx].partList[ip1].y - bList[bidx].partList[ip2].y;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    bList[bidx].partList[ip2].fx -= fx;
                    bList[bidx].partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[0]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[0]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength; //Considering the periodic boundary
                    const double ry = bList[bidx].partList[ip1].y - pbox[0]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[0]->partList[ip2].fx -= fx;
                    pbox[0]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[1]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[1]->partList[ip2].x;
                    const double ry = bList[bidx].partList[ip1].y - pbox[1]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[1]->partList[ip2].fx -= fx;
                    pbox[1]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[2]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[2]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = bList[bidx].partList[ip1].y - pbox[2]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[2]->partList[ip2].fx -= fx;
                    pbox[2]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[3]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[3]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength;
                    const double ry = bList[bidx].partList[ip1].y - pbox[3]->partList[ip2].y;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[3]->partList[ip2].fx -= fx;
                    pbox[3]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[4]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[4]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = bList[bidx].partList[ip1].y - pbox[4]->partList[ip2].y;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[4]->partList[ip2].fx -= fx;
                    pbox[4]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[5]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[5]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength;
                    const double ry = bList[bidx].partList[ip1].y - pbox[5]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[5]->partList[ip2].fx -= fx;
                    pbox[5]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[6]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[6]->partList[ip2].x;
                    const double ry = bList[bidx].partList[ip1].y - pbox[6]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[6]->partList[ip2].fx -= fx;
                    pbox[6]->partList[ip2].fy -= fy;
                }
                for (uint64_t ip2 = 0; ip2 < pbox[7]->NumParticles; ip2++)
                {
                    const double rx = bList[bidx].partList[ip1].x - pbox[7]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = bList[bidx].partList[ip1].y - pbox[7]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    double fx, fy;
                    forceFunction(rx, ry, &fx, &fy);
                    bList[bidx].partList[ip1].fx += fx;
                    bList[bidx].partList[ip1].fy += fy;
                    pbox[7]->partList[ip2].fx -= fx;
                    pbox[7]->partList[ip2].fy -= fy;
                }
            }
        }
    }
#endif

//Forces from the wall
#ifdef WithXWall
    if (pxid == 0)
    {
        for (uint64_t by = 0; by < YLocalSize; by++)
        {
            uint64_t bidx = by * XLocalSize;
            for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
            {
                if (bList[bidx].partList[pidx].x < 0)
                    bList[bidx].partList[pidx].fx += wallForce(bList[bidx].partList[pidx].x - LWall * BoxLength);
            }
        }
    }
    else if (pyid == NPy - 1)
    {
        for (uint64_t by = 0; by < YLocalSize; by++)
        {
            uint64_t bidx = by * XLocalSize + XLocalSize - 1;
            for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
            {
                if (bList[bidx].partList[pidx].x > XLength)
                    bList[bidx].partList[pidx].fx += wallForce(bList[bidx].partList[pidx].x - RWall * BoxLength);
            }
        }
    }
#endif
#ifdef WithYWalls
    if (pyid == 0)
    {
        for (uint64_t bx = 0; bx < XLocalSize; bx++)
        {
            uint64_t bidx = bx;
            for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
            {
                if (bList[bidx].partList[pidx].y < 0)
                    bList[bidx].partList[pidx].fy += wallForce(bList[bidx].partList[pidx].y - UWall * BoxLength);
            }
        }
    }
    else if (pyid == NPy - 1)
    {
        for (uint64_t bx = 0; bx < XLocalSize; bx++)
        {
            uint64_t bidx = (YLocalSize - 1) * XLocalSize + bx;
            for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
            {
                if (bList[bidx].partList[pidx].y > YLength)
                    bList[bidx].partList[pidx].fy += wallForce(bList[bidx].partList[pidx].y - DWall * BoxLength);
            }
        }
    }
#endif
    Vec4d pPos(probe_l->x - PlateRadius, probe_l->x + PlateRadius, probe_l->y - PlateRadius, probe_l->y + PlateRadius);
    Vec4d bBox = floor(pPos / BoxLength);
    probe_l->fx = 0;
    probe_l->fy = 0;
    int32_t bx1 = bBox[0] - pxid * XLocalSize;
    int32_t bx2 = bBox[1] - pxid * XLocalSize;
    int32_t by1 = bBox[2] - pyid * YLocalSize;
    int32_t by2 = bBox[3] - pyid * YLocalSize;
    //Only search when the object overlapped with the region simulated by the processor
    if (bx1 < (int32_t)XLocalSize && bx2 >= 0 && by1 < (int32_t)YLocalSize && by2 >= 0)
    {
        //Determine the boxes to search
        if (bx2 >= (int32_t)XLocalSize)
            bx2 = (int32_t)XLocalSize - 1;
        if (bx1 < 0)
            bx1 = 0;
        if (by2 >= (int32_t)YLocalSize)
            by2 = (int32_t)YLocalSize - 1;
        if (by1 < 0)
            by1 = 0;
        for (uint64_t by = by1; by <= by2; by++)
        {
            for (uint64_t bx = bx1; bx <= bx2; bx++)
            {
                uint64_t bidx = by * XLocalSize + bx;
                for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
                {
                    const double x = bList[bidx].partList[pidx].x - probe_l->x;
                    const double y = bList[bidx].partList[pidx].y - probe_l->y;

                    double pos2 = x * x + y * y;
                    if (pos2 < PlateRadius * PlateRadius)
                    {
                        //double pos=PlateRadius-sqrt(pos2);
                        //double force=10.0;
                        double r = sqrt(pos2) - (PlateRadius - ObjWallCutoff);
                        double r2 = r * r;
                        double r4 = r2 * r2;
                        double r6 = r2 * r4;
                        double r7 = r2 * r4 * r;
                        double rdVdr = 24.0 * Objk * ObjSigma6 * (2.0 * ObjSigma6 / r6 - 1.0) / r7 / sqrt(pos2);
                        double fx = x * rdVdr;
                        double fy = y * rdVdr;
                        //double fx=force * x;
                        //double fy=force * y;

                        probe_l->fx -= fx;
                        probe_l->fy -= fy;
                        bList[bidx].partList[pidx].fx += fx;
                        bList[bidx].partList[pidx].fy += fy;
                    }
                }
            }
        }
    }
    double recfx = 0, recfy = 0;
    MPI_Allreduce(&(probe_l->fx), &recfx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&(probe_l->fy), &recfy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    probe_r->fx = recfx;
    probe_r->fy = recfy + probe_r->k * (PlateNatPos - probe_r->y);
    Vec4d pPos2(probe_r->x - PlateRadius, probe_r->x + PlateRadius, probe_r->y - PlateRadius, probe_r->y + PlateRadius);
    bBox = floor(pPos2 / BoxLength);
    probe_r->fx = 0;
    probe_r->fy = 0;
    bx1 = bBox[0] - pxid * XLocalSize;
    bx2 = bBox[1] - pxid * XLocalSize;
    by1 = bBox[2] - pyid * YLocalSize;
    by2 = bBox[3] - pyid * YLocalSize;
    //Only search when the object overlapped with the region simulated by the processor
    if (bx1 < (int32_t)XLocalSize && bx2 >= 0 && by1 < (int32_t)YLocalSize && by2 >= 0)
    {
        //Determine the boxes to search
        if (bx2 >= (int32_t)XLocalSize)
            bx2 = (int32_t)XLocalSize - 1;
        if (bx1 < 0)
            bx1 = 0;
        if (by2 >= (int32_t)YLocalSize)
            by2 = (int32_t)YLocalSize - 1;
        if (by1 < 0)
            by1 = 0;
        for (uint64_t by = by1; by <= by2; by++)
        {
            for (uint64_t bx = bx1; bx <= bx2; bx++)
            {
                uint64_t bidx = by * XLocalSize + bx;
                for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
                {
                    const double x = bList[bidx].partList[pidx].x - probe_r->x;
                    const double y = bList[bidx].partList[pidx].y - probe_r->y;

                    double pos2 = x * x + y * y;
                    if (pos2 < PlateRadius * PlateRadius)
                    {
                        //double pos=PlateRadius-sqrt(pos2);
                        //double force=10.0;
                        double r = sqrt(pos2) - (PlateRadius - ObjWallCutoff);
                        double r2 = r * r;
                        double r4 = r2 * r2;
                        double r6 = r2 * r4;
                        double r7 = r2 * r4 * r;
                        double rdVdr = 24.0 * Objk * ObjSigma6 * (2.0 * ObjSigma6 / r6 - 1.0) / r7 / sqrt(pos2);
                        double fx = x * rdVdr;
                        double fy = y * rdVdr;
                        //double fx=force * x;
                        //double fy=force * y;

                        probe_r->fx -= fx;
                        probe_r->fy -= fy;
                        bList[bidx].partList[pidx].fx += fx;
                        bList[bidx].partList[pidx].fy += fy;
                    }
                }
            }
        }
    }
    recfx = 0;
    recfy = 0;
    MPI_Allreduce(&(probe_r->fx), &recfx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&(probe_r->fy), &recfy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    probe_r->fx = recfx;
    probe_r->fy = recfy + probe_r->k * (PlateNatPos - probe_r->y);
}

#ifdef QuorumSensing
void updateVelocities(box *__restrict__ bList, const uint64_t steps, const uint64_t NParticlesLocal,
                      box *__restrict__ UpperBoundary, box *__restrict__ LowerBoundary,
                      box *__restrict__ LeftBoundary, box *__restrict__ RightBoundary)
{
    int ibufH = 0;
    //Serializated buffer for particle number
    memset(bufferHuge_p, 0, NParticles * sizeof(uint32_t));
    uint32_t pidx = 0;
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;
    for (int32_t by = 0; by < YLocalSize; ++by)
    {
        for (int32_t bx = 0; bx < XLocalSize; ++bx)
        {

            box *pbox[9];
            for (int i = 0; i < 9; ++i)
                pbox[i] = &EmptyBox;

            //Neighbour boxes are labeled as
            // 0 1 2
            // 3 4 5
            // 6 7 8
            //Notice: in the words
            //the y-axis is taken to be downwards
            //the x-axis is taken to be rightwards

            if (by == 0)
            {
#ifdef WithYWalls
                if (pyid != 0)
#endif
                {
                    pbox[0] = UpperBoundary + bx;
                    pbox[1] = UpperBoundary + bx + 1;
                    pbox[2] = UpperBoundary + bx + 2;
                }
            }
            else
            {
                if (bx != 0)
                    pbox[0] = bList + bx + (by - 1) * XLocalSize - 1;
                pbox[1] = bList + bx + (by - 1) * XLocalSize;
                if (bx != XLocalSize - 1)
                    pbox[2] = bList + bx + (by - 1) * XLocalSize + 1;
            }

            if (bx == 0)
            {
#ifdef WithXWalls
                if (pxid != 0)
#endif
                {
                    if (by != 0)
                        pbox[0] = LeftBoundary + by - 1;
                    pbox[3] = LeftBoundary + by;
                    if (by != YLocalSize - 1)
                        pbox[6] = LeftBoundary + by + 1;
                }
            }
            else
            {
                pbox[3] = bList + bx - 1 + by * XLocalSize;
            }

            uint64_t bidx = by * XLocalSize + bx;
            pbox[4] = bList + bidx;

            if (bx == XLocalSize - 1)
            {
#ifdef WithXWalls
                if (pxid != NPx - 1)
#endif
                {
                    if (by != 0)
                        pbox[2] = RightBoundary + by - 1;
                    pbox[5] = RightBoundary + by;
                    if (by != YLocalSize - 1)
                        pbox[8] = RightBoundary + by + 1;
                }
            }
            else
            {
                pbox[5] = bList + bx + 1 + by * XLocalSize;
            }

            if (by == YLocalSize - 1)
            {
#ifdef WithYWalls
                if (pyid != NPy - 1)
#endif
                {
                    pbox[6] = LowerBoundary + bx;
                    pbox[7] = LowerBoundary + bx + 1;
                    pbox[8] = LowerBoundary + bx + 2;
                }
            }
            else
            {
                if (bx != 0)
                    pbox[6] = bList + bx + (by + 1) * XLocalSize - 1;
                pbox[7] = bList + bx + (by + 1) * XLocalSize;
                if (bx != XLocalSize - 1)
                    pbox[8] = bList + bx + (by + 1) * XLocalSize + 1;
            }

            for (uint64_t ip1 = 0; ip1 < bList[bidx].NumParticles; ip1++, pidx++)
            {
                double ux, uy;
                sincos(bList[bidx].partList[ip1].theta, &uy, &ux);
                //const double ux = cos(bList[bidx].partList[ip1].theta);
                //const double uy = sqrt(1.0 - ux * ux) * (2 * (bList[bidx].partList[ip1].theta < PI) - 1);
                for (uint64_t ip2 = 0; ip2 < pbox[0]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[0]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[0]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[1]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[1]->partList[ip2].x;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[1]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[2]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[2]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[2]->partList[ip2].y + (by == 0) * (pyid == 0) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[3]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[3]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[3]->partList[ip2].y;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[4]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[4]->partList[ip2].x;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[4]->partList[ip2].y;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[5]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[5]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[5]->partList[ip2].y;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[6]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[6]->partList[ip2].x + (bx == 0) * (pxid == 0) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[6]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[7]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[7]->partList[ip2].x;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[7]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
                for (uint64_t ip2 = 0; ip2 < pbox[8]->NumParticles; ip2++)
                {
                    const double rx = Epsilon * ux + bList[bidx].partList[ip1].x - pbox[8]->partList[ip2].x - (bx == XLocalSize - 1) * (pxid == NPx - 1) * XLength;
                    const double ry = Epsilon * uy + bList[bidx].partList[ip1].y - pbox[8]->partList[ip2].y - (by == YLocalSize - 1) * (pyid == NPy - 1) * YLength;
                    const double r2 = rx * rx + ry * ry;
                    bufferHuge_p[pidx] += (r2 < (R0 * R0));
                }
            }
        }
    }
    for (size_t ip = 0; ip < NParticlesLocal; ip++)
    {
        //Hat-top kernal
        bufferSmall[ip] = (double)bufferHuge_p[ip] * (1.0 / (PI * R0 * R0));
    }
    setQSVelocityArray(bufferSmall, NParticlesLocal);

    uint64_t i = 0;
    for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
    {
        for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
        {
            bList[bidx].partList[pidx].v = bufferSmall[i];
            ++i;
        }
    }
}
#endif

void initialization(box **__restrict__ bList, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r)
{
    *bList = (box *)malloc(NLocalBoxes * sizeof(box));

    probe_l->x = XLength * 0.5;
    probe_l->y = YThickness * BoxLength + PlateRadius + PlateShift;
    probe_l->vx = 0;
    probe_l->vy = 0;
    probe_l->fx = 0;
    probe_l->fy = 0;
    probe_l->k = PlateK;

    probe_r->x = XLength * 0.5;
    probe_r->y = YLength - (YThickness * BoxLength + PlateRadius + PlateShift);
    probe_r->vx = 0;
    probe_r->vy = 0;
    probe_r->fx = 0;
    probe_r->fy = 0;
    probe_r->k = PlateK;
    box *cbList = *bList;
    for (int i = 0; i < NLocalBoxes; i++)
    {
        cbList[i].partList = (particle *)calloc(setBoxCapacity, sizeof(particle));
        cbList[i].Capacity = setBoxCapacity;
        cbList[i].NumParticles = 0;
    }
}

void initialBoxB(box *boundary, const int size)
{
    for (int i = 0; i < size; i++)
    {
        boundary[i].partList = (particle *)calloc(setBoxCapacity, sizeof(particle));
        boundary[i].NumParticles = 0;
        boundary[i].Capacity = setBoxCapacity;
    }
}

//Put particles into hexagonal lattice if it's PFAPs
//Avoid the object. In the region [XThickness, XSize-XThickness] X [YThickness, YSize-YThickness]
//put the particles with density rhoG. Outside the region, put the particles with density rhoL.
//The processors with pxid=0 is responsible for calculating particles positions,
//and distribute them to other processors.
//Not sure if it's working now.
void putParticles(box *__restrict__ bList, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r, uint64_t *__restrict__ NParticlesLocal)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;
    *NParticlesLocal=0;
    if (pxid == 0)
    {
        const int8_t True = 1;
        const int8_t False = 0;
        MPI_Status mpistatus;

        double lboundary = pyid * YLocalSize * BoxLength;
        double rboundary = (pyid + 1) * YLocalSize * BoxLength;
        double llim = YThickness * BoxLength;
        double rlim = YLength - YThickness * BoxLength;
        double treshold = RhoL;

        uint64_t totalNumParticles = 0;
        //Calculate the local particle number
        if (rboundary <= llim || lboundary >= rlim)
        {
            totalNumParticles = RhoL * XLength * YLocalSize * BoxLength;
        }
        else if (lboundary <= llim)
        {
            double coffset = rboundary - llim;
            double area = coffset * (XSize - 2 * XThickness) * BoxLength;
            totalNumParticles = RhoL * (XLength * YLocalSize * BoxLength - area) + RhoG * area;
        }
        else if (rboundary >= rlim)
        {
            double coffset = rlim - lboundary;
            double area = coffset * (XSize - 2 * XThickness) * BoxLength;
            totalNumParticles = RhoL * (XLength * YLocalSize * BoxLength - area) + RhoG * area;
        }
        else
        {
            double area = YLocalSize * BoxLength * (XSize - 2 * XThickness) * BoxLength;
            totalNumParticles = RhoL * (XLength * YLocalSize * BoxLength - area) + RhoG * area;
        }
        uint64_t rejected = 0;

#ifdef PairwiseForces
        const double LatticeSize = Sigma * 0.93;
        const int32_t NXSites = floor(XLength / LatticeSize);
        int32_t previousLines = floor(YLocalSize * BoxLength * (pyid) / (0.86602540378443864676 * LatticeSize) - 0.5) + 1;
        if (previousLines < 0)
            previousLines = 0;
        const int32_t NYSites = floor(YLocalSize * BoxLength * (pyid + 1) / (0.86602540378443864676 * LatticeSize) - 0.5) + 1 - previousLines;
        bool *IfOccupied = (bool *)calloc(NXSites * NYSites, sizeof(bool));
#endif

        for (uint64_t i = 0; i < totalNumParticles; i++)
        {
            //bool object = 1;
            particle tempP;
#ifdef QuorumSensing
            bool ifInside = 0;
            //do
            //{
            tempP.x = genrand64_real3() * XLength;
            tempP.y = genrand64_real3() * (YLength/NPy);
            double realx = tempP.x;
            double realy = tempP.y + pyid * (YLength/NPy);
            ifInside = (realx - probe_l->x) * (realx - probe_l->x) + (realy - probe_l->y) * (realy - probe_l->y) <= PlateRadius * PlateRadius;
            ifInside = ifInside || ((realx - probe_r->x) * (realx - probe_r->x) + (realy - probe_r->y) * (realy - probe_r->y) <= PlateRadius * PlateRadius);
            //} while (ifInside);
            if (ifInside)
            {
                ++rejected;
                continue;
            }
            tempP.x = realx;
            tempP.y = realy;

            tempP.theta = genrand64_real2() * (2.0 * PI);
            tempP.fx = 0;
            tempP.fy = 0;
#endif

#ifdef PairwiseForces
            uint32_t ix, iy, is;
            bool ifInside = 0;
            do
            {
                ix = (uint32_t)floor(genrand64_real3() * NXSites);
                iy = (uint32_t)floor(genrand64_real3() * NYSites);
                is = iy * NXSites + ix;
            } while (IfOccupied[is]);
            IfOccupied[is] = 1;
            tempP.x = ((iy + previousLines) & 1) * (0.5 * LatticeSize) + ix * LatticeSize;
            tempP.y = (iy + previousLines + 0.5) * (0.86602540378443864676 * LatticeSize);
            ifInside = (tempP.x - probe_l->x) * (tempP.x - probe_l->x) + (tempP.y - probe_l->y) * (tempP.y - probe_l->y) <= PlateRadius * PlateRadius;
            ifInside = ifInside || ((tempP.x - probe_r->x) * (tempP.x - probe_r->x) + (tempP.y - probe_r->y) * (tempP.y - probe_r->y) <= PlateRadius * PlateRadius);
            if (ifInside)
            {
                ++rejected;
                continue;
            }
            tempP.theta = genrand64_real2() * (2.0 * PI);
            tempP.fx = 0;
            tempP.fy = 0;
#endif

#ifdef RTP
            tempP.nextTumbTime = -log(genrand64_real3()) * TauT;
#endif

            int32_t bx = (int32_t)floor(tempP.x / BoxLength);
            int32_t pxid_dest = bx / XLocalSize;
            if (pxid_dest == 0)
            {
                int32_t by = (uint32_t)floor(tempP.y / BoxLength) - pyid * YLocalSize;
                uint64_t bidx = by * XLocalSize + bx;
                addParticleInBox(bList, tempP, bidx);
                ++*NParticlesLocal;
            }
            else
            {
                MPI_Send(&True, 1, MPI_INT8_T, pyid * NPx + pxid_dest, pxid_dest, MPI_COMM_WORLD);
                MPI_Send(&tempP, 1, MPI_PARTICLE, pyid * NPx + pxid_dest, pxid_dest, MPI_COMM_WORLD);
            }
        }
#ifdef PairwiseForces
        free(IfOccupied);
#endif
        for (int i = 1; i < NPx; ++i)
            MPI_Send(&False, 1, MPI_INT8_T, pyid * NPx + i, i, MPI_COMM_WORLD);
    }
    else
    {
        int8_t ifreceiving = 1;
        MPI_Status mpistatus;
        MPI_Recv(&ifreceiving, 1, MPI_INT8_T, pyid * NPx, pxid, MPI_COMM_WORLD, &mpistatus);
        while (ifreceiving == 1)
        {
            particle tempP;
            MPI_Recv(&tempP, 1, MPI_PARTICLE, pyid * NPx, pxid, MPI_COMM_WORLD, &mpistatus);
            int32_t bx = floor(tempP.x / BoxLength) - pxid * XLocalSize;
            int32_t by = floor(tempP.y / BoxLength) - pyid * YLocalSize;
            uint64_t bidx = by * XLocalSize + bx;
            addParticleInBox(bList, tempP, bidx);
            ++*NParticlesLocal;
            MPI_Recv(&ifreceiving, 1, MPI_INT8_T, pyid * NPx, pxid, MPI_COMM_WORLD, &mpistatus);
        }
    }
}

// #ifdef CircularInitialCondition
// void putParticlesCircular(box *__restrict__ bList, uint64_t *__resctrict__ NParticlesLocal)
// {
//     int pid = 0;
//     MPI_Comm_rank(MPI_COMM_WORLD, &pid);

//     double lboundary = pid * YLocalSize * BoxLength;
//     double rboundary = (pid + 1) * YLocalSize * BoxLength;
//     double centery = (NProcesses / 2) * YLocalSize * BoxLength;
//     double llim = centery - Radius;
//     double rlim = centery + Radius;
//     double treshold = RhoL;
//     if (rboundary <= llim || lboundary >= rlim)
//     {
//         *NParticlesLocal = RhoG * XLength * YLocalSize * BoxLength;
//         treshold = RhoG;
//     }
//     else if (lboundary <= llim)
//     {
//         double coffset = centery - rboundary;
//         double theta = acos(coffset / Radius);
//         double area = Radius * (Radius * theta - coffset * sin(theta));
//         *NParticlesLocal = RhoG * (XLength * YLocalSize * BoxLength - area) + RhoL * area;
//     }
//     else if (rboundary >= rlim)
//     {
//         double coffset = lboundary - centery;
//         double theta = acos(coffset / Radius);
//         double area = Radius * (Radius * theta - coffset * sin(theta));
//         *NParticlesLocal = RhoG * (XLength * YLocalSize * BoxLength - area) + RhoL * area;
//     }
//     else if (lboundary < centery)
//     {
//         double coffset1 = centery - lboundary;
//         double coffset2 = centery - rboundary;
//         double theta1 = acos(coffset1 / Radius);
//         double theta2 = acos(coffset2 / Radius);
//         double area = Radius * (Radius * (theta2 - theta1) - coffset2 * sin(theta2) + coffset1 * sin(theta1));
//         *NParticlesLocal = RhoG * (XLength * YLocalSize * BoxLength - area) + RhoL * area;
//     }
//     else
//     {
//         double coffset1 = rboundary - centery;
//         double coffset2 = lboundary - centery;
//         double theta1 = acos(coffset1 / Radius);
//         double theta2 = acos(coffset2 / Radius);
//         double area = Radius * (Radius * (theta2 - theta1) - coffset2 * sin(theta2) + coffset1 * sin(theta1));
//         *NParticlesLocal = RhoG * (XLength * YLocalSize * BoxLength - area) + RhoL * area;
//     }
//     for (uint64_t i = 0; i < *NParticlesLocal; i++)
//     {
//         bool object = 1;
//         particle tempP;
// #ifdef QuorumSensing
//         do
//         {
//             tempP.x = genrand64_real3() * XLength;
//             tempP.y = genrand64_real3() * (YLength / NProcesses);
//             double sqrdist = (tempP.y + lboundary - centery) * (tempP.y + lboundary - centery) + (tempP.x - XLength * 0.5) * (tempP.x - XLength * 0.5);
//             double r = genrand64_real3() * treshold;
//             bool ifOutside = sqrdist > (Radius * Radius);
//             object = r > (RhoG * ifOutside + RhoL * (!ifOutside));
//         } while (object);

//         uint32_t bx = (uint32_t)floor(tempP.x / BoxLength);
//         uint32_t by = (uint32_t)floor(tempP.y / BoxLength);
//         tempP.y += pid * YLength / NProcesses;
//         uint32_t bidx = by * XSize + bx;
//         tempP.theta = genrand64_real2() * (2.0 * PI);
//         tempP.fx = 0;
//         tempP.fy = 0;
// #endif

// #ifdef PairwiseForces
//         uint32_t bx, by, bidx;
//         do
//         {
//             do
//             {
//                 bx = (uint32_t)floor(genrand64_real3() * XSize);
//                 by = (uint32_t)floor(genrand64_real3() * YLocalSize);
//                 bidx = by * XSize + bx;
//             } while (bList[bidx].NumParticles > 1);
//             bool config = (bx & 1) ^ (by & 1);
//             if (config)
//             {
//                 tempP.x = (bx + 0.15 + bList[bidx].NumParticles * 0.7) * BoxLength;
//                 tempP.y = (by + pid * YLocalSize + 0.5) * BoxLength;
//             }
//             else
//             {
//                 tempP.x = (bx + 0.5) * BoxLength;
//                 tempP.y = (by + pid * YLocalSize + 0.15 + bList[bidx].NumParticles * 0.7) * BoxLength;
//             }
//             double sqrdist = (tempP.y - centery) * (tempP.y - centery) + (tempP.x - XLength * 0.5) * (tempP.x - XLength * 0.5);
//             double r = genrand64_real3() * treshold;
//             bool ifOutside = sqrdist > Radius * Radius;
//             object = r > (RhoG * ifOutside + RhoL * (!ifOutside));
//         } while (object);
//         tempP.theta = genrand64_real2() * (2.0 * PI);
//         tempP.fx = 0;
//         tempP.fy = 0;
// #endif

// #ifdef RTP
//         tempP.nextTumbTime = -log(genrand64_real3()) * TauT;
// #endif
//         addParticleInBox(bList, tempP, bidx);
//     }
// }
// #endif

void updateBoundaryInformation(box *__restrict__ bList,
                               box *__restrict__ UpperBoundary, box *__restrict__ LowerBoundary,
                               box *__restrict__ LeftBoundary, box *__restrict__ RightBoundary)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

    const int lpxid = (pxid - 1 + NPx) % NPx;
    const int rpxid = (pxid + 1 + NPx) % NPx;
    const int upyid = (pyid - 1 + NPy) % NPy;
    const int dpyid = (pyid + 1 + NPy) % NPy;

    const int upid = upyid * NPx + pxid; //up
    const int dpid = dpyid * NPx + pxid; //down
    const int lpid = pyid * NPx + lpxid; //left
    const int rpid = pyid * NPx + rpxid; //right

    const int ulpid = upyid * NPx + lpxid; //up-left
    const int urpid = upyid * NPx + rpxid; //up-right
    const int dlpid = dpyid * NPx + lpxid; //down-left
    const int drpid = dpyid * NPx + rpxid; //down-right

    //Firstly, update informations from up-left and down-left processors

    MPI_Request reqs[4]; // required variable for non-blocking calls
    MPI_Status stats[4]; // required variable for Waitall routine

    // post non-blocking receives and sends for neighbors
    MPI_Irecv(&(UpperBoundary[0].NumParticles), 1, MPI_UINT64_T, ulpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&(LowerBoundary[0].NumParticles), 1, MPI_UINT64_T, dlpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend(&(bList[NLocalBoxes - 1].NumParticles), 1, MPI_UINT64_T, drpid, drpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&(bList[XLocalSize - 1].NumParticles), 1, MPI_UINT64_T, urpid, urpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    //Check if UpperBoundary and LowerBoundary have enough space
    if (UpperBoundary[0].Capacity < UpperBoundary[0].NumParticles)
    {
        particle *temp = (particle *)realloc((void *)UpperBoundary[0].partList, UpperBoundary[0].NumParticles * sizeof(particle));
        if (temp != NULL)
            UpperBoundary[0].partList = temp;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        UpperBoundary[0].Capacity = UpperBoundary[0].NumParticles;
    }
    if (LowerBoundary[0].Capacity < LowerBoundary[0].NumParticles)
    {
        particle *temp2 = (particle *)realloc((void *)LowerBoundary[0].partList, LowerBoundary[0].NumParticles * sizeof(particle));
        if (temp2 != NULL)
            LowerBoundary[0].partList = temp2;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        LowerBoundary[0].Capacity = LowerBoundary[0].NumParticles;
    }

    // post non-blocking receives and sends for neighbors
    MPI_Irecv((UpperBoundary[0].partList), UpperBoundary[0].NumParticles, MPI_PARTICLE, ulpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv((LowerBoundary[0].partList), LowerBoundary[0].NumParticles, MPI_PARTICLE, dlpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend((bList[NLocalBoxes - 1].partList), bList[NLocalBoxes - 1].NumParticles, MPI_PARTICLE, drpid, drpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend((bList[XLocalSize - 1].partList), bList[XLocalSize - 1].NumParticles, MPI_PARTICLE, urpid, urpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    //Secondly, update informations from up and down processors

    for (uint64_t i = 1; i <= XLocalSize; i++)
    {
        // post non-blocking receives and sends for neighbors
        MPI_Irecv(&(UpperBoundary[i].NumParticles), 1, MPI_UINT64_T, upid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&(LowerBoundary[i].NumParticles), 1, MPI_UINT64_T, dpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&(bList[NLocalBoxes - XLocalSize + i - 1].NumParticles), 1, MPI_UINT64_T, dpid, dpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&(bList[i - 1].NumParticles), 1, MPI_UINT64_T, upid, upid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        if (UpperBoundary[i].Capacity < UpperBoundary[i].NumParticles)
        {
            particle *temp = (particle *)realloc((void *)UpperBoundary[i].partList, UpperBoundary[i].NumParticles * sizeof(particle));
            if (temp != NULL)
                UpperBoundary[i].partList = temp;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            UpperBoundary[i].Capacity = UpperBoundary[i].NumParticles;
        }
        if (LowerBoundary[i].Capacity < LowerBoundary[i].NumParticles)
        {
            particle *temp2 = (particle *)realloc((void *)LowerBoundary[i].partList, LowerBoundary[i].NumParticles * sizeof(particle));
            if (temp2 != NULL)
                LowerBoundary[i].partList = temp2;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            LowerBoundary[i].Capacity = LowerBoundary[i].NumParticles;
        }

        // post non-blocking receives and sends for neighbors
        MPI_Irecv((UpperBoundary[i].partList), UpperBoundary[i].NumParticles, MPI_PARTICLE, upid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv((LowerBoundary[i].partList), LowerBoundary[i].NumParticles, MPI_PARTICLE, dpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend((bList[NLocalBoxes - XLocalSize + i - 1].partList), bList[NLocalBoxes - XLocalSize + i - 1].NumParticles, MPI_PARTICLE, dpid, dpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend((bList[i - 1].partList), bList[i - 1].NumParticles, MPI_PARTICLE, upid, upid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);
    }

    //Thirdly, update information from up-right and down-right processors

    // post non-blocking receives and sends for neighbors
    MPI_Irecv(&(UpperBoundary[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, urpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&(LowerBoundary[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, drpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend(&(bList[NLocalBoxes - XLocalSize].NumParticles), 1, MPI_UINT64_T, dlpid, dlpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&(bList[0].NumParticles), 1, MPI_UINT64_T, ulpid, ulpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    if (UpperBoundary[XLocalSize + 1].Capacity < UpperBoundary[XLocalSize + 1].NumParticles)
    {
        particle *temp = (particle *)realloc((void *)UpperBoundary[XLocalSize + 1].partList, UpperBoundary[XLocalSize + 1].NumParticles * sizeof(particle));
        if (temp != NULL)
            UpperBoundary[XLocalSize + 1].partList = temp;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        UpperBoundary[XLocalSize + 1].Capacity = UpperBoundary[XLocalSize + 1].NumParticles;
    }
    if (LowerBoundary[XLocalSize + 1].Capacity < LowerBoundary[XLocalSize + 1].NumParticles)
    {
        particle *temp2 = (particle *)realloc((void *)LowerBoundary[XLocalSize + 1].partList, LowerBoundary[XLocalSize + 1].NumParticles * sizeof(particle));
        if (temp2 != NULL)
            LowerBoundary[XLocalSize + 1].partList = temp2;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        LowerBoundary[XLocalSize + 1].Capacity = LowerBoundary[XLocalSize + 1].NumParticles;
    }

    // post non-blocking receives and sends for neighbors
    MPI_Irecv((UpperBoundary[XLocalSize + 1].partList), UpperBoundary[XLocalSize + 1].NumParticles, MPI_PARTICLE, urpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv((LowerBoundary[XLocalSize + 1].partList), LowerBoundary[XLocalSize + 1].NumParticles, MPI_PARTICLE, drpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend((bList[NLocalBoxes - XLocalSize].partList), bList[NLocalBoxes - XLocalSize].NumParticles, MPI_PARTICLE, dlpid, dlpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend((bList[0].partList), bList[0].NumParticles, MPI_PARTICLE, ulpid, ulpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    //Finally, update information from left and right processors

    for (uint64_t i = 0; i < YLocalSize; i++)
    {
        // post non-blocking receives and sends for neighbors
        MPI_Irecv(&(LeftBoundary[i].NumParticles), 1, MPI_UINT64_T, lpid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&(RightBoundary[i].NumParticles), 1, MPI_UINT64_T, rpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&(bList[(i + 1) * XLocalSize - 1].NumParticles), 1, MPI_UINT64_T, rpid, rpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&(bList[i * XLocalSize].NumParticles), 1, MPI_UINT64_T, lpid, lpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        if (LeftBoundary[i].Capacity < LeftBoundary[i].NumParticles)
        {
            particle *temp = (particle *)realloc((void *)LeftBoundary[i].partList, LeftBoundary[i].NumParticles * sizeof(particle));
            if (temp != NULL)
                LeftBoundary[i].partList = temp;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            LeftBoundary[i].Capacity = LeftBoundary[i].NumParticles;
        }
        if (RightBoundary[i].Capacity < RightBoundary[i].NumParticles)
        {
            particle *temp2 = (particle *)realloc((void *)RightBoundary[i].partList, RightBoundary[i].NumParticles * sizeof(particle));
            if (temp2 != NULL)
                RightBoundary[i].partList = temp2;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            RightBoundary[i].Capacity = RightBoundary[i].NumParticles;
        }

        // post non-blocking receives and sends for neighbors
        MPI_Irecv((LeftBoundary[i].partList), LeftBoundary[i].NumParticles, MPI_PARTICLE, lpid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv((RightBoundary[i].partList), RightBoundary[i].NumParticles, MPI_PARTICLE, rpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend((bList[(i + 1) * XLocalSize - 1].partList), bList[(i + 1) * XLocalSize - 1].NumParticles, MPI_PARTICLE, rpid, rpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend((bList[i * XLocalSize].partList), bList[i * XLocalSize].NumParticles, MPI_PARTICLE, lpid, lpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);
    }
}

void destruction(box *__restrict__ bList)
{
    for (int i = 0; i < NLocalBoxes; i++)
    {
        free(bList[i].partList);
    }
    free(bList);
#ifdef QuorumSensing
    free(bufferSmall);
    free(bufferHuge_p);
#endif
}

void step(box *__restrict__ bList, const uint64_t steps, uint64_t *__restrict__ NParticlesLocal,
          box *__restrict__ FirstLine, box *__restrict__ LastLine,
          box *__restrict__ FirstCol, box *__restrict__ LastCol, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

    //If you are afraid of accumulating random round-off errors, synchronize the probe information all the time.
    // if (steps >= TimeCritical * StepsPerSec)
    // {
    //     const double dt=1.0/StepsPerSec;
    //     double probe_l_dvx = probe_l->fx / PlateMass * dt;
    //     double probe_l_dvy = probe_l->fy / PlateMass * dt;
    //     probe_l->x += probe_l->vx * dt + probe_l_dvx * dt * 0.5;
    //     probe_l->y += probe_l->vy * dt + probe_l_dvy * dt * 0.5;
    //     probe_l->vx += probe_l_dvx;
    //     probe_l->vy += probe_l_dvy;

    //     double probe_r_dvx = probe_r->fx / PlateMass * dt;
    //     double probe_r_dvy = probe_r->fy / PlateMass * dt;
    //     probe_r->x += probe_r->vx * dt + probe_r_dvx * dt * 0.5;
    //     probe_r->y += probe_r->vy * dt + probe_r_dvy * dt * 0.5;
    //     probe_r->vx += probe_r_dvx;
    //     probe_r->vy += probe_r_dvy;

    //     double meanX = 0, meanY = 0;
    //     MPI_Allreduce(&probe_l->x, &meanX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     MPI_Allreduce(&probe_l->y, &meanY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     probe_l->x = meanX / NProcesses;
    //     probe_l->y = meanY / NProcesses;
    //     MPI_Allreduce(&probe_l->vx, &meanX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     MPI_Allreduce(&probe_l->vy, &meanY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     probe_l->vx = meanX / NProcesses;
    //     probe_l->vy = meanY / NProcesses;
    //     MPI_Allreduce(&probe_r->x, &meanX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     MPI_Allreduce(&probe_r->y, &meanY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     probe_r->x = meanX / NProcesses;
    //     probe_r->y = meanY / NProcesses;
    //     MPI_Allreduce(&probe_r->vx, &meanX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     MPI_Allreduce(&probe_r->vy, &meanY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //     probe_r->vx = meanX / NProcesses;
    //     probe_r->vy = meanY / NProcesses;
    // }

    if (steps>=TimeCritical*StepsPerSec)
    {
    	probe_l->y+=PlateV*(1.0/StepsPerSec);
    	probe_r->y-=PlateV*(1.0/StepsPerSec);
    }
    for (uint64_t i = 0; i < XLocalSize + 2; i++)
    {
        FirstLine[i].NumParticles = 0;
        LastLine[i].NumParticles = 0;
    }
    for (uint64_t i = 0; i < YLocalSize; i++)
    {
        FirstCol[i].NumParticles = 0;
        LastCol[i].NumParticles = 0;
    }
#ifdef ABP
    char flush = 8;
    Vec4d rand1(0);
    Vec4d rand2(0);
    Vec4d rrand1(0);
    Vec4d crand2(0);
    Vec4d srand2(0);
    double rdifangle[8];
    // bool flagRollDice = 1;
    // double rand1a, cosrand2a, rand2a, sinrand2a;
#endif
    for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
    {
        for (uint64_t i = 0; i < bList[bidx].NumParticles; i++)
        {
            double timeRemaining = 1.0 / StepsPerSec;
            double ddt = 1.0 / StepsPerSec;

#ifdef RTP
            bool flagTumb = 0;
            bool flagContinue = 1;
            while (flagContinue)
            {
                if (bList[bidx].partList[i].nextTumbTime > timeRemaining)
                {
                    ddt = timeRemaining;
                    bList[bidx].partList[i].nextTumbTime -= timeRemaining;
                    flagContinue = 0;
                }
                else
                {
                    ddt = bList[bidx].partList[i].nextTumbTime;
                    timeRemaining = timeRemaining - bList[bidx].partList[i].nextTumbTime;
                    flagTumb = 1;
                }
#endif

                //double dev = sqrt(2.0 * Dtrans * ddt);
                //double rand1 = sqrt(-2.0 * log(genrand64_real3()));
                //double rand2 = 2.0 * PI * genrand64_real3();
                //Vec2d temp(bList[bidx].partList[i].theta, rand2);
                Vec2d temp(bList[bidx].partList[i].theta, 0);
                Vec2d rs, rc; //s=sin, c=cos
                rs = sincos(&rc, temp);
#ifdef QuorumSensing
                bList[bidx].partList[i].x += (bList[bidx].partList[i].v * rc[0] + Mobility * bList[bidx].partList[i].fx) * ddt; // + rand1 * rc[1] * dev;
                bList[bidx].partList[i].y += (bList[bidx].partList[i].v * rs[0] + Mobility * bList[bidx].partList[i].fy) * ddt; // + rand1 * rs[1] * dev;
#endif
#ifdef PairwiseForces
                bList[bidx].partList[i].x += (v * rc[0] + Mobility * bList[bidx].partList[i].fx) * ddt; // + rand1 * rc[1] * dev;
                bList[bidx].partList[i].y += (v * rs[0] + Mobility * bList[bidx].partList[i].fy) * ddt; // + rand1 * rs[1] * dev;
#endif

#ifdef RTP
                if (flagTumb)
                {
                    bList[bidx].partList[i].theta = genrand64_real2() * (2.0 * PI);
                    bList[bidx].partList[i].nextTumbTime = -log(genrand64_real3()) * TauT;
                    flagTumb = 0;
                }
            }
#endif

#ifdef ABP
            double dev = sqrt(2.0 * Drot * timeRemaining);
            if (flush == 8)
            {
                for (int ir = 0; ir < 4; ir++)
                {
                    rand1.insert(ir, genrand64_real3());
                    rand2.insert(ir, genrand64_real3());
                }
                rrand1 = sqrt(-2.0 * log(rand1));
                rand2 *= (2.0 * PI);
                srand2 = sincos(&crand2, rand2);
                srand2 *= rrand1;
                crand2 *= rrand1;
                srand2 *= dev;
                crand2 *= dev;
                srand2.store(rdifangle);
                crand2.store(rdifangle + 4);
                flush = 0;
            }
            bList[bidx].partList[i].theta += rdifangle[flush];
            ++flush;
            // if (flagRollDice)
            // {
            //     rand1a = sqrt(-2.0 * log(genrand64_real3()));
            //     rand2a = 2.0 * PI * genrand64_real3();

            //     //cosrand2a = cos(rand2a);
            //     sincos(rand2a, &sinrand2a, &cosran2a);
            //     bList[bidx].partList[i].theta += rand1a * cosrand2a * dev;
            //     flagRollDice = 0;
            // }
            // else
            // {
            //     //double sinrand2a = sqrt(1.0 - cosrand2a * cosrand2a) * (2 * (rand2a < PI) - 1);
            //     bList[bidx].partList[i].theta += rand1a * sinrand2a * dev;
            //     flagRollDice = 1;
            // }
#endif
            bList[bidx].partList[i].fx = 0;
            bList[bidx].partList[i].fy = 0;
        }
    }

    for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
    {
        for (uint64_t i = 0; i < bList[bidx].NumParticles; i++)
        {
            Vec2d temp(bList[bidx].partList[i].x / BoxLength, bList[bidx].partList[i].y / BoxLength);
            Vec2d r = floor(temp);

            int32_t bx = r[0] - pxid * XLocalSize;
            int32_t by = r[1] - pyid * YLocalSize;
#ifdef WithXWalls
            if (pxid == 0 && bx < 0)
                bx = 0;
            else if (pxid == NPx - 1 && bx > (int32_t)XLocalSize - 1)
                bx = XLocalSize - 1;
#endif
#ifdef WithYWalls
            if (pyid == 0 && by < 0)
                by = 0;
            else if (pyid == NPy - 1 && by > (int32_t)YLocalSize - 1)
                by = YLocalSize - 1;
#endif

            //If we found a particle is outside the region of the current processor,
            //put it into the corresponding list to be exchanged.
            if (by < 0)
            {
#ifndef WithYWalls
                if (bList[bidx].partList[i].y < 0)
                    bList[bidx].partList[i].y += YSize * BoxLength;
#endif
#ifndef WithXWalls
                bList[bidx].partList[i].x -= floor(bList[bidx].partList[i].x / XLength) * XLength;
#endif
                if (FirstLine[bx + 1].NumParticles == FirstLine[bx + 1].Capacity)
                {
                    particle *temp = (particle *)realloc((void *)FirstLine[bx + 1].partList, FirstLine[bx + 1].Capacity * 2 * sizeof(particle));
                    if (temp != NULL)
                        FirstLine[bx + 1].partList = temp;
                    else
                    {
                        printf("ERROR: Not enough memory!\n");
                        return;
                    }
                    FirstLine[bx + 1].Capacity *= 2;
                }
                FirstLine[bx + 1].partList[FirstLine[bx + 1].NumParticles] = bList[bidx].partList[i];
                ++FirstLine[bx + 1].NumParticles;
                deleteParticleInBox(bList, bidx, i);
                --(*NParticlesLocal);
                --i;
                //This is necessary, otherwise we miss some particles to check.
                //Some particles may be checked multiple times, but the effect is neglectable.
            }
            else if (by >= YLocalSize)
            {
#ifndef WithYWalls
                if (bList[bidx].partList[i].y > YSize * BoxLength)
                    bList[bidx].partList[i].y -= YSize * BoxLength;
#endif
#ifndef WithXWalls
                bList[bidx].partList[i].x -= floor(bList[bidx].partList[i].x / XLength) * XLength;
#endif
                if (LastLine[bx + 1].NumParticles == LastLine[bx + 1].Capacity)
                {
                    particle *temp = (particle *)realloc((void *)LastLine[bx + 1].partList, LastLine[bx + 1].Capacity * 2 * sizeof(particle));
                    if (temp != NULL)
                        LastLine[bx + 1].partList = temp;
                    else
                    {
                        printf("ERROR: Not enough memory!\n");
                        return;
                    }
                    LastLine[bx + 1].Capacity *= 2;
                }
                LastLine[bx + 1].partList[LastLine[bx + 1].NumParticles] = bList[bidx].partList[i];
                ++LastLine[bx + 1].NumParticles;
                deleteParticleInBox(bList, bidx, i);
                --(*NParticlesLocal);
                --i;
            }
            else if (bx < 0)
            {
#ifndef WithXWalls
                if (bList[bidx].partList[i].x < 0)
                    bList[bidx].partList[i].x += XSize * BoxLength;
#endif
                if (FirstCol[by].NumParticles == FirstCol[by].Capacity)
                {
                    particle *temp = (particle *)realloc((void *)FirstCol[by].partList, FirstCol[by].Capacity * 2 * sizeof(particle));
                    if (temp != NULL)
                        FirstCol[by].partList = temp;
                    else
                    {
                        printf("ERROR: Not enough memory!\n");
                        return;
                    }
                    FirstCol[by].Capacity *= 2;
                }
                FirstCol[by].partList[FirstCol[by].NumParticles] = bList[bidx].partList[i];
                ++FirstCol[by].NumParticles;
                deleteParticleInBox(bList, bidx, i);
                --(*NParticlesLocal);
                --i;
            }
            else if (bx >= XLocalSize)
            {
#ifndef WithXWalls
                if (bList[bidx].partList[i].x > XSize * BoxLength)
                    bList[bidx].partList[i].x -= XSize * BoxLength;
#endif
                if (LastCol[by].NumParticles == LastCol[by].Capacity)
                {
                    particle *temp = (particle *)realloc((void *)LastCol[by].partList, LastCol[by].Capacity * 2 * sizeof(particle));
                    if (temp != NULL)
                        LastCol[by].partList = temp;
                    else
                    {
                        printf("ERROR: Not enough memory!\n");
                        return;
                    }
                    LastCol[by].Capacity *= 2;
                }
                LastCol[by].partList[LastCol[by].NumParticles] = bList[bidx].partList[i];
                ++LastCol[by].NumParticles;
                deleteParticleInBox(bList, bidx, i);
                --(*NParticlesLocal);
                --i;
            }
            else
            {
                uint64_t bidx_new = by * XLocalSize + bx;
                if (bidx != bidx_new)
                {
                    moveParticleInBox(bList, i, bidx, bidx_new);
                    --i;
                }
            }
        }
    }
}

//Things are almost the same as updateBoundaryInformation()
void exchangeParticles(box *__restrict__ bList, uint64_t *__restrict__ NParticlesLocal,
                       box *__restrict__ UpperBoundary, box *__restrict__ LowerBoundary,
                       box *__restrict__ LeftBoundary, box *__restrict__ RightBoundary,
                       box *__restrict__ FirstLine, box *__restrict__ LastLine,
                       box *__restrict__ FirstCol, box *__restrict__ LastCol)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

    const int lpxid = (pxid - 1 + NPx) % NPx;
    const int rpxid = (pxid + 1 + NPx) % NPx;
    const int upyid = (pyid - 1 + NPy) % NPy;
    const int dpyid = (pyid + 1 + NPy) % NPy;

    const int upid = upyid * NPx + pxid;
    const int dpid = dpyid * NPx + pxid;
    const int lpid = pyid * NPx + lpxid;
    const int rpid = pyid * NPx + rpxid;

    const int ulpid = upyid * NPx + lpxid;
    const int urpid = upyid * NPx + rpxid;
    const int dlpid = dpyid * NPx + lpxid;
    const int drpid = dpyid * NPx + rpxid;

    MPI_Request reqs[4]; // required variable for non-blocking calls
    MPI_Status stats[4]; // required variable for Waitall routine

    // post non-blocking receives and sends for neighbors
    MPI_Irecv(&(UpperBoundary[0].NumParticles), 1, MPI_UINT64_T, ulpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&(LowerBoundary[0].NumParticles), 1, MPI_UINT64_T, dlpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend(&(LastLine[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, drpid, drpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&(FirstLine[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, urpid, urpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    if (UpperBoundary[0].Capacity < UpperBoundary[0].NumParticles)
    {
        particle *temp = (particle *)realloc((void *)UpperBoundary[0].partList, UpperBoundary[0].NumParticles * sizeof(particle));
        if (temp != NULL)
            UpperBoundary[0].partList = temp;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        UpperBoundary[0].Capacity = UpperBoundary[0].NumParticles;
    }
    if (LowerBoundary[0].Capacity < LowerBoundary[0].NumParticles)
    {
        particle *temp2 = (particle *)realloc((void *)LowerBoundary[0].partList, LowerBoundary[0].NumParticles * sizeof(particle));
        if (temp2 != NULL)
            LowerBoundary[0].partList = temp2;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        LowerBoundary[0].Capacity = LowerBoundary[0].NumParticles;
    }

    // post non-blocking receives and sends for neighbors
    MPI_Irecv((UpperBoundary[0].partList), UpperBoundary[0].NumParticles, MPI_PARTICLE, ulpid, pid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv((LowerBoundary[0].partList), LowerBoundary[0].NumParticles, MPI_PARTICLE, dlpid, pid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend((LastLine[XLocalSize + 1].partList), LastLine[XLocalSize + 1].NumParticles, MPI_PARTICLE, drpid, drpid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend((FirstLine[XLocalSize + 1].partList), FirstLine[XLocalSize + 1].NumParticles, MPI_PARTICLE, urpid, urpid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    for (uint64_t j = 0; j < UpperBoundary[0].NumParticles; ++j)
    {
        addParticleInBox(bList, UpperBoundary[0].partList[j], 0);
    }
    (*NParticlesLocal) += UpperBoundary[0].NumParticles;
    for (uint64_t j = 0; j < LowerBoundary[0].NumParticles; ++j)
    {
        addParticleInBox(bList, LowerBoundary[0].partList[j], NLocalBoxes - XLocalSize);
    }
    (*NParticlesLocal) += LowerBoundary[0].NumParticles;

    for (uint64_t i = 1; i < XLocalSize + 1; i++)
    {

        // post non-blocking receives and sends for neighbors
        MPI_Irecv(&(UpperBoundary[i].NumParticles), 1, MPI_UINT64_T, upid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&(LowerBoundary[i].NumParticles), 1, MPI_UINT64_T, dpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&(LastLine[i].NumParticles), 1, MPI_UINT64_T, dpid, dpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&(FirstLine[i].NumParticles), 1, MPI_UINT64_T, upid, upid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        if (UpperBoundary[i].Capacity < UpperBoundary[i].NumParticles)
        {
            particle *temp = (particle *)realloc((void *)UpperBoundary[i].partList, UpperBoundary[i].NumParticles * sizeof(particle));
            if (temp != NULL)
                UpperBoundary[i].partList = temp;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            UpperBoundary[i].Capacity = UpperBoundary[i].NumParticles;
        }
        if (LowerBoundary[i].Capacity < LowerBoundary[i].NumParticles)
        {
            particle *temp2 = (particle *)realloc((void *)LowerBoundary[i].partList, LowerBoundary[i].NumParticles * sizeof(particle));
            if (temp2 != NULL)
                LowerBoundary[i].partList = temp2;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            LowerBoundary[i].Capacity = LowerBoundary[i].NumParticles;
        }

        // post non-blocking receives and sends for neighbors
        MPI_Irecv((UpperBoundary[i].partList), UpperBoundary[i].NumParticles, MPI_PARTICLE, upid, pid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv((LowerBoundary[i].partList), LowerBoundary[i].NumParticles, MPI_PARTICLE, dpid, pid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend((LastLine[i].partList), LastLine[i].NumParticles, MPI_PARTICLE, dpid, dpid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend((FirstLine[i].partList), FirstLine[i].NumParticles, MPI_PARTICLE, upid, upid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        for (uint64_t j = 0; j < UpperBoundary[i].NumParticles; ++j)
        {
            addParticleInBox(bList, UpperBoundary[i].partList[j], i - 1);
        }
        (*NParticlesLocal) += UpperBoundary[i].NumParticles;
        for (uint64_t j = 0; j < LowerBoundary[i].NumParticles; ++j)
        {
            addParticleInBox(bList, LowerBoundary[i].partList[j], NLocalBoxes - XLocalSize + i - 1);
        }
        (*NParticlesLocal) += LowerBoundary[i].NumParticles;
    }

    // post non-blocking receives and sends for neighbors
    MPI_Irecv(&(UpperBoundary[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, urpid, pid, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&(LowerBoundary[XLocalSize + 1].NumParticles), 1, MPI_UINT64_T, drpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend(&(LastLine[0].NumParticles), 1, MPI_UINT64_T, dlpid, dlpid, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&(FirstLine[0].NumParticles), 1, MPI_UINT64_T, ulpid, ulpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    if (UpperBoundary[XLocalSize + 1].Capacity < UpperBoundary[XLocalSize + 1].NumParticles)
    {
        particle *temp = (particle *)realloc((void *)UpperBoundary[XLocalSize + 1].partList, UpperBoundary[XLocalSize + 1].NumParticles * sizeof(particle));
        if (temp != NULL)
            UpperBoundary[XLocalSize + 1].partList = temp;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        UpperBoundary[XLocalSize + 1].Capacity = UpperBoundary[XLocalSize + 1].NumParticles;
    }
    if (LowerBoundary[XLocalSize + 1].Capacity < LowerBoundary[XLocalSize + 1].NumParticles)
    {
        particle *temp2 = (particle *)realloc((void *)LowerBoundary[XLocalSize + 1].partList, LowerBoundary[XLocalSize + 1].NumParticles * sizeof(particle));
        if (temp2 != NULL)
            LowerBoundary[XLocalSize + 1].partList = temp2;
        else
        {
            printf("ERROR: Not enough memory!\n");
            return;
        }
        LowerBoundary[XLocalSize + 1].Capacity = LowerBoundary[XLocalSize + 1].NumParticles;
    }

    // post non-blocking receives and sends for neighbors
    MPI_Irecv((UpperBoundary[XLocalSize + 1].partList), UpperBoundary[XLocalSize + 1].NumParticles, MPI_PARTICLE, urpid, pid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv((LowerBoundary[XLocalSize + 1].partList), LowerBoundary[XLocalSize + 1].NumParticles, MPI_PARTICLE, drpid, pid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[1]);

    MPI_Isend((LastLine[0].partList), LastLine[0].NumParticles, MPI_PARTICLE, dlpid, dlpid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend((FirstLine[0].partList), FirstLine[0].NumParticles, MPI_PARTICLE, ulpid, ulpid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[3]);

    // wait for all non-blocking operations to complete
    MPI_Waitall(4, reqs, stats);

    for (uint64_t j = 0; j < UpperBoundary[XLocalSize + 1].NumParticles; ++j)
    {
        addParticleInBox(bList, UpperBoundary[XLocalSize + 1].partList[j], XLocalSize - 1);
    }
    (*NParticlesLocal) += UpperBoundary[XLocalSize + 1].NumParticles;
    for (uint64_t j = 0; j < LowerBoundary[XLocalSize + 1].NumParticles; ++j)
    {
        addParticleInBox(bList, LowerBoundary[XLocalSize + 1].partList[j], NLocalBoxes - 1);
    }
    (*NParticlesLocal) += LowerBoundary[XLocalSize + 1].NumParticles;

    for (uint64_t i = 0; i < YLocalSize; i++)
    {

        // post non-blocking receives and sends for neighbors
        MPI_Irecv(&(LeftBoundary[i].NumParticles), 1, MPI_UINT64_T, lpid, pid, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&(RightBoundary[i].NumParticles), 1, MPI_UINT64_T, rpid, pid + NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&(LastCol[i].NumParticles), 1, MPI_UINT64_T, rpid, rpid, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&(FirstCol[i].NumParticles), 1, MPI_UINT64_T, lpid, lpid + NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        if (LeftBoundary[i].Capacity < LeftBoundary[i].NumParticles)
        {
            particle *temp = (particle *)realloc((void *)LeftBoundary[i].partList, LeftBoundary[i].NumParticles * sizeof(particle));
            if (temp != NULL)
                LeftBoundary[i].partList = temp;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            LeftBoundary[i].Capacity = LeftBoundary[i].NumParticles;
        }
        if (RightBoundary[i].Capacity < RightBoundary[i].NumParticles)
        {
            particle *temp2 = (particle *)realloc((void *)RightBoundary[i].partList, RightBoundary[i].NumParticles * sizeof(particle));
            if (temp2 != NULL)
                RightBoundary[i].partList = temp2;
            else
            {
                printf("ERROR: Not enough memory!\n");
                return;
            }
            RightBoundary[i].Capacity = RightBoundary[i].NumParticles;
        }

        // post non-blocking receives and sends for neighbors
        MPI_Irecv((LeftBoundary[i].partList), LeftBoundary[i].NumParticles, MPI_PARTICLE, lpid, pid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv((RightBoundary[i].partList), RightBoundary[i].NumParticles, MPI_PARTICLE, rpid, pid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend((LastCol[i].partList), LastCol[i].NumParticles, MPI_PARTICLE, rpid, rpid + 2 * NProcesses, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend((FirstCol[i].partList), FirstCol[i].NumParticles, MPI_PARTICLE, lpid, lpid + 3 * NProcesses, MPI_COMM_WORLD, &reqs[3]);

        // wait for all non-blocking operations to complete
        MPI_Waitall(4, reqs, stats);

        for (uint64_t j = 0; j < LeftBoundary[i].NumParticles; ++j)
        {
            addParticleInBox(bList, LeftBoundary[i].partList[j], i * XLocalSize);
        }
        (*NParticlesLocal) += LeftBoundary[i].NumParticles;
        for (uint64_t j = 0; j < RightBoundary[i].NumParticles; ++j)
        {
            addParticleInBox(bList, RightBoundary[i].partList[j], (i + 1) * XLocalSize - 1);
        }
        (*NParticlesLocal) += RightBoundary[i].NumParticles;
    }
}

void collectingResult(box *__restrict__ bList, particle *__restrict__ pList)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Status status;
    uint64_t *pNum = (uint64_t *)calloc(NBoxes, sizeof(uint64_t));
    if (pid != 0)
    {
        for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
        {
            MPI_Ssend(&bList[bidx].NumParticles, 1, MPI_UINT64_T, 0, pid * NLocalBoxes + bidx, MPI_COMM_WORLD);
        }
    }
    else
    {
        for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
        {
            for (int ic = 1; ic < NProcesses; ic++)
            {
                MPI_Recv(&pNum[ic * NLocalBoxes + bidx], 1, MPI_UINT64_T, ic, ic * NLocalBoxes + bidx, MPI_COMM_WORLD, &status);
            }
        }
    }

    if (pid != 0)
    {
        for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
        {
            MPI_Ssend(bList[bidx].partList, bList[bidx].NumParticles, MPI_PARTICLE, 0, NBoxes + pid * NLocalBoxes + bidx, MPI_COMM_WORLD);
        }
    }
    else
    {
        uint64_t offset = 0; //pNum[NLocalBoxes];
        for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
        {
            for (int ic = 1; ic < NProcesses; ic++)
            {
                MPI_Recv((void *)(pList + offset), pNum[ic * NLocalBoxes + bidx], MPI_PARTICLE, ic, NBoxes + ic * NLocalBoxes + bidx, MPI_COMM_WORLD, &status);
                offset += pNum[ic * NLocalBoxes + bidx];
            }
        }
        uint64_t i = 0;
        for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
        {
            for (uint64_t pidx = 0; pidx < bList[bidx].NumParticles; pidx++)
            {
                *(pList + offset + i) = bList[bidx].partList[pidx];
                ++i;
            }
        }
    }
    free(pNum);
}

void Save(box *__restrict__ bList, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r, uint64_t NParticlesLocal)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char filename[10];
    char buffer[5];
    sprintf(buffer, "%02d", pid);
    strcpy(filename, "sav");
    strcat(filename, buffer);
    strcat(filename, ".bin");
    FILE *pOutputFile = fopen(filename, "wb");

    fwrite(probe_l, sizeof(probePlate), 1, pOutputFile);
    fwrite(probe_r, sizeof(probePlate), 1, pOutputFile);
    fwrite(&NParticlesLocal, sizeof(uint64_t), 1, pOutputFile);

    for (uint64_t bidx = 0; bidx < NLocalBoxes; bidx++)
    {
        fwrite(bList[bidx].partList, sizeof(particle), bList[bidx].NumParticles, pOutputFile);
    }
    fclose(pOutputFile);
}

void Load(box *__restrict__ bList, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r, uint64_t *pNParticlesLocal)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

    char filename[10];
    char buffer[5];
    sprintf(buffer, "%02d", pid);
    strcpy(filename, "sav");
    strcat(filename, buffer);
    strcat(filename, ".bin");
    FILE *pInputFile = fopen(filename, "rb");

    fread(probe_l, sizeof(probePlate), 1, pInputFile);
    fread(probe_r, sizeof(probePlate), 1, pInputFile);
    fread(pNParticlesLocal, sizeof(uint64_t), 1, pInputFile);
    //printf("%d %lld\n",pid,*pNParticlesLocal);
    for (uint64_t pidx = 0; pidx < *pNParticlesLocal; pidx++)
    {
        particle temp;
        temp.fx=0;
	temp.fy=0;
	fread(&temp, sizeof(particle), 1, pInputFile);
        int32_t bx = (int32_t)floor(temp.x / BoxLength) - pxid * XLocalSize;
        int32_t by = (int32_t)floor(temp.y / BoxLength) - pyid * YLocalSize;
        if (bx < 0 && pxid == 0)
            bx = 0;
        if (bx > (int32_t)XLocalSize - 1 && pxid == NPx - 1)
            bx = XLocalSize - 1;
        if (by < 0 && pyid == 0)
            by = 0;
        if (by > (int32_t)YLocalSize - 1 && pyid == NPy - 1)
            by = YLocalSize - 1;
        uint32_t bidx = by * XLocalSize + bx;
        addParticleInBox(bList, temp, bidx);
    }
    fclose(pInputFile);
}

void LoadOld(box *__restrict__ bList, probePlate *__restrict__ probe_l, probePlate *__restrict__ probe_r, uint64_t *pNParticlesLocal)
{
    int pid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const int pxid = pid % NPx;
    const int pyid = pid / NPx;

    char filename[10];
    char buffer[5];
    sprintf(buffer, "%02d", pid);
    strcpy(filename, "sav");
    strcat(filename, buffer);
    strcat(filename, ".bin");
    FILE *pInputFile = fopen(filename, "rb");

    struct
    {
        double x;
        double y;
        double fx;
        double fy;
        double k;
    } oldProbe;

    fread(&oldProbe, sizeof(oldProbe), 1, pInputFile);
    probe_l->x=oldProbe.x;
    probe_l->y=oldProbe.y;
    probe_l->vx=0;
    probe_l->vy=0;
    probe_l->fx=0;
    probe_l->fy=0;
    probe_l->k=PlateK;

    fread(&oldProbe, sizeof(oldProbe), 1, pInputFile);
    probe_r->x=oldProbe.x;
    probe_r->y=oldProbe.y;
    probe_r->vx=0;
    probe_r->vy=0;
    probe_r->fx=0;
    probe_r->fy=0;
    probe_r->k=PlateK;

    fread(pNParticlesLocal, sizeof(uint64_t), 1, pInputFile);
    //printf("%d %lld\n",pid,*pNParticlesLocal);
    for (uint64_t pidx = 0; pidx < *pNParticlesLocal; pidx++)
    {
        particle temp;
        fread(&temp, sizeof(particle), 1, pInputFile);
        int32_t bx = (int32_t)floor(temp.x / BoxLength) - pxid * XLocalSize;
        int32_t by = (int32_t)floor(temp.y / BoxLength) - pyid * YLocalSize;
        if (bx < 0 && pxid == 0)
            bx = 0;
        if (bx > (int32_t)XLocalSize - 1 && pxid == NPx - 1)
            bx = XLocalSize - 1;
        if (by < 0 && pyid == 0)
            by = 0;
        if (by > (int32_t)YLocalSize - 1 && pyid == NPy - 1)
            by = YLocalSize - 1;
        uint32_t bidx = by * XLocalSize + bx;
        addParticleInBox(bList, temp, bidx);
    }
    fclose(pInputFile);
}
#endif

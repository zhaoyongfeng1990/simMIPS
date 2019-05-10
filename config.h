#define QuorumSensing // QuorumSensing or PairwiseForces
//#define PairwiseForces
#define ABP // For active Brownian particles
//#define RTP // For run-and-tumble particles

#ifdef PairwiseForces
//#define WCAForces // WCAForces or HarmonicForces
#define HarmonicForces // WCAForces or HarmonicForces
#endif

#define WithYWalls
//#define CircularInitialCondition

#define RhoG 9.21613
#define RhoL 91.9605
uint64_t NParticles = 0;

#ifdef CircularInitialCondition
#define Radius 40
#else
//If initial configuration is rectangular, the thickness of the boundary
#define YThickness 64UL
#define XThickness 0
#endif

//Number of parallelised grids
//Imaging we have a matrix of processors, they are the dimensions of the matrix.
#define NPx 4
#define NPy 12

//Number of processes
#define NProcesses (NPx * NPy)

#define XSize 128UL //System size, in the unit of BoxLength
#define YSize 384UL
//Size of the system in the local process
#define XLocalSize (XSize / NPx)
#define YLocalSize (YSize / NPy)
#define setBoxCapacity 16 //Initial capacity in each box

#define StepsPerSec 128UL
#define TotalTime 2100
#define TimeCritical 100
#define TotalSteps (TotalTime * StepsPerSec)

#define Mobility 1
#define V 10.0
#define Dtrans 0 //Translational diffusion constant

#define WallK 1e-4 //Strengh of the wall potential
//The reference position in calculating the force from the wall
//y-axis is downwards, x-axis is rightwards
#define XWall 1
#define LWall (-XWall)
#define RWall (XSize + XWall)
#define UWall (-XWall)
#define DWall (YSize + XWall)

//Parameters related to the object
#define PlateRadius 8.0
#define PlateMass 100.0
#define PlateNatPos (YSize * BoxLength * 0.5) //Natural postion of the spring
#define PlateK 0                              //Spring constant
#define PlateShift 0.0                        //Shift from the liquid-gas interface
#define PlateV 0.04                            //Velocity

#ifdef RTP
#define TauT 1.0
#endif

#ifdef ABP
#define Drot 1.0
#endif

//Parameters for the wall potential on the object
#define Objk (1e-4)
#define ObjWallCutoff 5.0
#define ObjSigma6 (ObjWallCutoff * ObjWallCutoff * ObjWallCutoff * ObjWallCutoff * ObjWallCutoff * ObjWallCutoff * 0.5)

//Parameters for WCA potential of PFAPs
#ifdef WCAForces
#define Sigma 1.0 //Interaction range
#define Sigma2 (Sigma * Sigma)
#define Sigma6 (Sigma2 * Sigma2 * Sigma2)
#define BoxLength (Sigma * 1.122462048309373) //Box size
#define InteractionEpsilon 1.0
#endif

#ifdef HarmonicForces
#define Sigma 1.0 //Interaction range
#define Sigma2 (Sigma*Sigma)
#define BoxLength (Sigma)
#define InteractionEpsilon 100.0
#endif

//Parameters for the QSAPs
#ifdef QuorumSensing
#define R0 1.0                  //0.722926280210    //Quorum sensing range
#define Epsilon 0.0             //Quorum sensing bias
#define Zinv 2.1435657748759676 //To be calculated numerically by Mathematica
#define BoxLength (R0 + Epsilon)
#define InteractionK 2.0
#define V0 10
#define V1 1
#define Rhom 40.0
#define Lf 10.0
#endif

//#define BoxLength Sigma > (R0 + Epsilon) ? Sigma : (R0 + Epsilon)

//Size of the system
#define XLength (XSize * BoxLength)
#define YLength (YSize * BoxLength)
#define BoxLength2 (BoxLength * BoxLength)

//Number of boxes
#define NBoxes (XSize * YSize)
#define NLocalBoxes (XLocalSize * YLocalSize)

#define PI 3.14159265358979323846264338327950288419716939937510

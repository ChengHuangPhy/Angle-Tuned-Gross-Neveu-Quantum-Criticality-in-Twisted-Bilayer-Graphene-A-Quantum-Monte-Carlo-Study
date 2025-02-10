#include "head.h"
////////////////////////////// Control parameters
const int L=6,//Linear size
          ntau=200,//Number of imaginary time slices
          mc_sweep=800,//Total Monte Carlo sweeps
          mc_thermal=400;//Thermalization sweeps
const double T=1.0/2.0,//Temperature in meV
             theta=1.08/180.0*pi,//Tuned angle
             optrate=0.4;//Accepting rate
///////////////////////////////
const int nb=2,
	  nk=L*L,
	  LM=nk*nb,
	  NM=LM*LM,
	  NM1=(NM+LM)/2,
          Nspr=nk*4;

const double q_unit=8.0*pi*sin(theta/2.0)/(3.0*sqrt(3.0)*a),
             Omega=3.0*sqrt(3.0)*nk*a*a*0.125/pow(sin(theta*0.5),2),
             nk2=2.0/(nk*nk);

double square(double x);
double abs1(double x);
void BM_eigen(vector<double> G,double *k,double *w,MKL_Complex16 *vr,double theta);
void zmatsvd(MKL_Complex16 *A,double *S,MKL_Complex16 *U,MKL_Complex16 *V,int LM);
void zmatmul0(MKL_Complex16 a,MKL_Complex16 *A,MKL_Complex16 *A1,int NM);
void zmatmul1(double *S,MKL_Complex16 *A,int LM);
void _zmatmul2(MKL_Complex16 *A,double *S,MKL_Complex16 *A1,int LM);
void zmatmul2(MKL_Complex16 *A,double *S,int LM);
void zmatmul(MKL_Complex16 *A,MKL_Complex16 *A1,MKL_Complex16 *A2,int LM);
void zmatcp(MKL_Complex16 *A,MKL_Complex16 *A1,int NM);
void hmatcp(MKL_Complex16 *A1,MKL_Complex16 *A2,int NM1);
void hmatcpr(MKL_Complex16 *A1,MKL_Complex16 *A,int LM);
void zmatcp1(MKL_Complex16 *A,MKL_Complex16 *A1,int NM);
void hmatcpr1(MKL_Complex16 *A1,MKL_Complex16 *A,int LM);
void zmati(MKL_Complex16 *A,int LM);
void zmatadd(MKL_Complex16 *A,MKL_Complex16 *A1,int NM);
void zmatsubs(MKL_Complex16 *A,MKL_Complex16 *A1,int NM);
void zmatinit(MKL_Complex16 *A,int LM,int NM);
void zmat_I(MKL_Complex16 *A,int LM,int NM);
void zmataddI(MKL_Complex16 *A,int LM,int NM);
MKL_Complex16 zmatdet(MKL_Complex16 *A,int LM,int NM);
void zmatmaxmin(double *A,double *A1,int LM);
void eAh(MKL_Complex16 *A,MKL_Complex16 *A1,MKL_Complex16 *eARN,int LM,int NM);
double Vx(double x,double q_unit);
void Mat(int iqG,double val,MKL_Complex16 *Mmn,MKL_Complex16 *_Mmn,int *minusqG,int nqG,vector<int> kkqG,int nk,int LM,int NM);
void SVD0(double *phi,MKL_Complex16 *sprMmn,MKL_Complex16 *UV,double *S,int nqG,MKL_Complex16 *B,MKL_Complex16 *Bi,
	  double *tE,MKL_Complex16 *B1,int *ispr,int *tspr,int ntau,int LM,int NM,int NM1,int Nspr);
void Gtt(int itau,MKL_Complex16 *gtt,MKL_Complex16 *gt0,MKL_Complex16 *UV,double *S,double *DL,double *DR,
	 MKL_Complex16 *RM,MKL_Complex16 *RN,MKL_Complex16 *eARM,MKL_Complex16 *eARN,int LM,int NM);
void SVD1(int itau,MKL_Complex16 *B,MKL_Complex16 *UV,double *S,int direc,MKL_Complex16 *B1,
          MKL_Complex16 *RM,MKL_Complex16 *RN,int ntau,int LM,int NM);
void zmattran(MKL_Complex16 *A,MKL_Complex16 *A1,int LM,int NM);
double SIVC(MKL_Complex16 *gtt,MKL_Complex16 *RM,int nk,int LM,int NM);

int main(){

int rank,size;
MPI_Init(0,0);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);

std::random_device seed;
std::ranlux48 engine(seed());
std::uniform_int_distribution<unsigned> uni(0,10000);
std::normal_distribution<double> distribution(0.0,1.0);

MKL_Complex16 eARM[NM],
              eARN[NM],
	      RM[NM],
	      RN[NM];

double S[(ntau+1)*LM],
       DR[2*LM],
       DL[2*LM],
       erp=1.0e-1,
       derp=1.0e-1,
       val[8],
       k0[nk*2],
       q0[nk*2],
       nearestq[7*2];
int Bool,
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9;

for(i0=0;i0<L;i0++)
    for(i1=0;i1<L;i1++)
        for(i2=0;i2<2;i2++){
            k0[(i0*L+i1)*2+i2]=G0[0][i2]/L*i0+G0[1][i2]/L*i1;
            q0[(i0*L+i1)*2+i2]=G0[0][i2]/L*i0+G0[1][i2]/L*i1;
	}

vector<double> G;
for(i0=-10;i0<11;i0++)
    for(int i1=-10;i1<11;i1++){
	for(i2=0;i2<2;i2++)
	    val[i2]=G0[0][i2]*i0+G0[1][i2]*i1;
	if(square(val[0])+square(val[1])<BMcut){
	    G.push_back(val[0]);
	    G.push_back(val[1]);
	}
    }
int nG=G.size()/2,
    LBM=nG*4,
    NBM=LBM*LBM;

vector<double> qG,
	       MqG;
vector<int> qqG;
for(i0=0;i0<nk;i0++)//q+G
    for(i1=0;i1<nG;i1++){
	for(i2=0;i2<2;i2++)
	    val[i2]=q0[i0*2+i2]+G[i1*2+i2];
	val[2]=square(val[0])+square(val[1]);
	if(val[2]>1.0e-8&&val[2]<3.0001){
	    qG.push_back(val[0]);
	    qG.push_back(val[1]);
	    MqG.push_back(sqrt(val[2]));
	    qqG.push_back(i0);
	}
    }
int nqG=MqG.size();

int *minusqG=new int[nqG];
for(i0=0;i0<nqG;i0++){
    Bool=1;
    for(i1=0;i1<nqG;i1++)
	if(abs1(qG[i0*2]+qG[i1*2])<1.0e-8&&abs1(qG[i0*2+1]+qG[i1*2+1])<1.0e-8){
	    minusqG[i0]=i1;
	    Bool=0;
	    break;
	}
    if(Bool){
	cout<<"minusqG error"<<endl;
	exit(0);
    }
}

vector<int> kkqG,
	    GkqG;
for(i0=0;i0<nk;i0++)//k
    for(i1=0;i1<nqG;i1++){//qG
	for(i2=0;i2<2;i2++)
	    val[i2]=k0[i0*2+i2]+qG[i1*2+i2];
	Bool=1;
	for(i2=0;i2<nk;i2++){
	    for(i3=0;i3<nG;i3++)
		if(abs1(val[0]-G[i3*2]-k0[i2*2])<1.0e-8&&abs1(val[1]-G[i3*2+1]-k0[i2*2+1])<1.0e-8){
		    kkqG.push_back(i2);
		    GkqG.push_back(i3);
		    Bool=0;
		    break;
		}
	    if(Bool==0)
		break;
	}
        if(Bool){
	    cout<<"kqG error"<<endl;
	    exit(0);
	}
    }

MKL_Complex16 *_Mmn=new MKL_Complex16[nk*nqG*4],
	      *Mmn=new MKL_Complex16[2*NM],
	      *sprMmn=new MKL_Complex16[nqG*Nspr];
int *ispr=new int[nqG*Nspr],
    *tspr=new int[nqG];
double *tE=new double[LM+2];
if(rank==0){
    MKL_Complex16 *vr=new MKL_Complex16[nk*LBM*nb];

    for(i0=0;i0<nk;i0++)
        BM_eigen(G,k0+i0*2,tE+i0*2,vr+i0*LBM*nb,theta);

    int *IG=new int[nG*nG];//IG indicates the shift of G
    for(i0=0;i0<nG;i0++)
        for(i1=0;i1<nG;i1++){
	    for(i2=0;i2<2;i2++)
	        val[i2]=G[i0*2+i2]+G[i1*2+i2];
            Bool=1;
            for(i2=0;i2<nG;i2++)
	        if(abs1(val[0]-G[i2*2])<1.0e-8&&abs1(val[1]-G[i2*2+1])<1.0e-8){
	            IG[i0*nG+i1]=i2;
		    Bool=0;
		    break;
	        }
	    if(Bool)
	        IG[i0*nG+i1]=-1;
        }

    for(i0=0;i0<nk;i0++)
        for(i1=0;i1<nqG;i1++)
            for(i2=0;i2<nb;i2++)
	        for(i3=0;i3<nb;i3++){
		    i4=(i0*nqG+i1)*4+i2*2+i3;
                    _Mmn[i4]=c0;
		    for(i5=0;i5<nG;i5++){
		        i6=IG[GkqG[i0*nqG+i1]*nG+i5];
		        if(i6>-1)
		            for(i7=0;i7<4;i7++){
		                i8=i0*LBM*nb+(i5*4+i7)*nb+i2;
		                i9=kkqG[i0*nqG+i1]*LBM*nb+(i6*4+i7)*nb+i3;
		                _Mmn[i4].real+=vr[i8].real*vr[i9].real+vr[i8].imag*vr[i9].imag;
		                _Mmn[i4].imag+=vr[i8].real*vr[i9].imag-vr[i8].imag*vr[i9].real;
			    }
		    }
		}
    delete[] IG;
    delete[] vr;
}
G.clear();
G.shrink_to_fit();
GkqG.clear();
GkqG.shrink_to_fit();
MPI_Bcast(tE,LM,MPI_DOUBLE,0,MPI_COMM_WORLD);
MPI_Bcast(_Mmn,nk*nqG*4,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

int htran[NM];
i0=0;
for(i1=0;i1<LM;i1++)
    for(i2=0;i2<=i1;i2++){
        htran[i1*LM+i2]=i0;
	i0++;
    }

double kBT=1.0e-3*e*T,
       dtau=1.0/kBT/ntau;
int CqG=0;
for(i0=0;i0<nqG;i0++)if((abs1(qG[i0*2+1])<1.0e-8&&qG[i0*2]>1.0e-8)||qG[i0*2+1]<-1.0e-8){
    Mat(i0,sqrt(dtau*Vx(MqG[i0],q_unit)*0.5/Omega),Mmn,_Mmn,minusqG,nqG,kkqG,nk,LM,NM);
    for(i1=0;i1<2;i1++){
	i2=CqG*2+i1;
	i3=0;
	for(i4=0;i4<LM;i4++)
	    for(i5=0;i5<=i4;i5++){
	        i6=i1*NM+i4*LM+i5;
	        if(abs1(Mmn[i6].real)>1.0e-8||abs1(Mmn[i6].imag)>1.0e-8){
		    sprMmn[i2*Nspr+i3]=Mmn[i6];
		    ispr[i2*Nspr+i3]=htran[i4*LM+i5];
		    i3++;
	        }
	    }
	tspr[i2]=i3;
    }
    CqG++;
}
delete[] _Mmn;
delete[] Mmn;
qG.clear();
qG.shrink_to_fit();
MqG.clear();
MqG.shrink_to_fit();
qqG.clear();
qqG.shrink_to_fit();
kkqG.clear();
kkqG.shrink_to_fit();

int NumAF=ntau*nqG,
    idr,
    itau,
    itauNM;

double *phi=new double[NumAF],
       *phiP=NULL,
       *dphi=new double[nqG],
       Sq=0.0,
       rateaccept;

MKL_Complex16 gtt[NM],
	      gt0[NM],
              *B=new MKL_Complex16[ntau*NM],
              *B1=new MKL_Complex16[ntau*NM1],
              *Bi=new MKL_Complex16[ntau*NM],
              *UV=new MKL_Complex16[(ntau+1)*2*NM],
	      *fgt0=new MKL_Complex16[(ntau+1)*LM],
	      weight[2],
	      valz,
	      _M[NM+NM1],
	      *MP=NULL;

for(i0=0;i0<NumAF;i0++)
    phi[i0]=distribution(engine);

for(i0=0;i0<LM;i0++)
    S[ntau*LM+i0]=1.0;
for(i0=0;i0<2;i0++)
   zmatinit(UV+ntau*2*NM+i0*NM,LM,NM);

for(i0=0;i0<(ntau+1)*LM;i0++)
    fgt0[i0]=c0;

for(i0=0;i0<LM;i0++)
    tE[i0]=exp(-dtau*tE[i0]*e);

SVD0(phi,sprMmn,UV,S,nqG,B,Bi,tE,B1,ispr,tspr,ntau,LM,NM,NM1,Nspr);

for(int imc=0;imc<mc_sweep;imc++){
    if(imc%50==0)
	cout<<"rank "<<rank<<" imc "<<imc<<endl;
    MPI_Barrier(MPI_COMM_WORLD);
    rateaccept=0.0;
    
    for(idr=1;idr>-1;idr--)
        for(itau=idr*(ntau-1);itau!=(1-idr)*ntau-idr;itau+=1-idr*2){
	    Gtt(itau+1,gtt,gt0,UV,S,DL,DR,RM,RN,eARM,eARN,LM,NM);
            if(imc>=mc_thermal){
	        if((itau+1)%50==0)
		    Sq+=SIVC(gtt,RM,nk,LM,NM)*nk2;
	        for(i0=0;i0<nk;i0++)
	            for(i1=0;i1<2;i1++){
		        i2=(itau+1)*LM+i0*2+i1;
		        i3=i0*nk*4+i1*LM+i0*2+i1;
		        fgt0[i2].real+=gt0[i3].real;
	                if(itau==ntau-1)
		            fgt0[i0*2+i1].real+=gtt[i3].real;
		    }
	    }
            itauNM=itau*NM;
	    phiP=phi+itau*nqG;
	    MP=B1+itau*NM1;
	    zmatcp(B+itauNM,_M,NM);
	    hmatcp(MP,_M+NM,NM1);
	    val[0]=0.0;
            for(i0=0;i0<nqG;i0++){
                dphi[i0]=distribution(engine)*erp;
	        val[0]-=(phiP[i0]+0.5*dphi[i0])*dphi[i0];
	    }
            for(i0=0;i0<nqG;i0++){
	        i1=i0*Nspr;
	        for(i2=0;i2<tspr[i0];i2++){
		    i3=ispr[i1];
		    MP[i3].real+=dphi[i0]*sprMmn[i1].real;
		    MP[i3].imag+=dphi[i0]*sprMmn[i1].imag;
		    i1++;
	        }
	    }
	    hmatcpr(MP,RN,LM);
	    eAh(RN,B+itauNM,eARN,LM,NM);
	    zmatmul(B+itauNM,Bi+itauNM,RN,LM);
	    zmat_I(RN,LM,NM);
	    zmatcp1(gtt,eARN,NM);
	    zmataddI(eARN,LM,NM);
	    zmatmul(RN,eARN,eARM,LM);
	    zmataddI(eARM,LM,NM);
	    valz=zmatdet(eARM,LM,NM);
	    val[0]=exp(val[0])*square(square(valz.real)+square(valz.imag));
	    if(val[0]>1.0)
	        rateaccept+=1.0;
	    else
	        rateaccept+=val[0];
            if(uni(engine)*1.0e-4<=val[0]){
	        zmatmul2(B+itauNM,tE,LM);
                for(i0=0;i0<nqG;i0++)
	            phiP[i0]+=dphi[i0];
	        hmatcpr1(MP,RN,LM);
	        eAh(RN,Bi+itauNM,eARN,LM,NM);
            }
            else{
	        zmatcp(_M,B+itauNM,NM);
	        hmatcp(_M+NM,MP,NM1);
	    }
            SVD1(itau,B+itauNM,UV,S,idr,B+itauNM+NM,RM,RN,ntau,LM,NM);
        }
    erp+=(rateaccept/ntau*0.5-optrate)*derp;
}

delete[] B;
delete[] B1;
delete[] Bi;
delete[] UV;
delete[] sprMmn;
delete[] ispr;
delete[] tspr;
delete[] tE;
delete[] minusqG;
delete[] phi;
delete[] dphi;

std::ofstream outf;

val[0]=0.5/(mc_sweep-mc_thermal);
for(i0=0;i0<(ntau+1)*LM;i0++){
    fgt0[i0].real*=val[0];
    fgt0[i0].imag*=val[0];
}
MKL_Complex16 *Fgt0=NULL;
if(rank==0)
    Fgt0=new MKL_Complex16[(size+1)*(ntau+1)*LM];
MPI_Gather(fgt0,(ntau+1)*LM,MPI_C_DOUBLE_COMPLEX,Fgt0,(ntau+1)*LM,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
if(rank==0){
    outf.open("data.dat",ios::out);
    for(i0=0;i0<size*(ntau+1)*LM;i0++)
	outf<<Fgt0[i0].real<<" "<<Fgt0[i0].imag<<" ";
    outf<<endl;
    outf.close();
}
MPI_Reduce(fgt0,Fgt0,(ntau+1)*LM,MPI_C_DOUBLE_COMPLEX,MPI_SUM,0,MPI_COMM_WORLD);
for(i0=0;i0<(ntau+1)*LM;i0++){
    fgt0[i0].real=fgt0[i0].real*fgt0[i0].real;
    fgt0[i0].imag=0.0;
}
MPI_Reduce(fgt0,Fgt0+(ntau+1)*LM,(ntau+1)*LM,MPI_C_DOUBLE_COMPLEX,MPI_SUM,0,MPI_COMM_WORLD);
delete[] fgt0;

Sq/=(mc_sweep-mc_thermal)*ntau/50*2;
MPI_Reduce(&Sq,val,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
Sq=Sq*Sq;
MPI_Reduce(&Sq,val+1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

if(rank==0){
    outf.open("data.dat",ios::app);
    outf<<"kBT (meV)  "<<kBT/(1.0e-3*e)<<endl;
    for(i0=0;i0<ntau+1;i0++){
        outf<<"itau "<<i0<<endl;
        for(i1=0;i1<LM;i1++){
            i2=i0*LM+i1;
            Fgt0[i2].real/=size;
            outf<<Fgt0[i2].real<<" "<<sqrt((Fgt0[(ntau+1)*LM+i2].real-pow(Fgt0[i2].real,2)*size)/size)<<endl;
        }
    }
    val[0]/=size;
    outf<<val[0]<<" "<<sqrt((val[1]-square(val[0])*size)/size);
    outf.close();
}

delete[] Fgt0;

MPI_Finalize();
return 0;
}

#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

int np_NX;
int np_NY;
int nx;
int ny;
int nz;

#define GOT_NORTH(rank) ((rank) > np_NX + 1)
#define GOT_EAST(rank)  ((rank) % (np_NX) != 0)
#define GOT_SOUTH(rank) ((rank - 1) / (np_NX) < np_NY - 1)
#define GOT_WEST(rank)  ((rank - 1) % np_NX != 0)

__global__ void jacobi3D_kernel(    
    float* data_in,
    float* data_out,
    float* halo_north, 
    float* halo_east, 
    float* halo_south, 
    float* halo_west, 
    float* halo_ne, 
    float* halo_se, 
    float* halo_sw, 
    float* halo_nw,
    int nx,
    int ny,
    int nz,
    int NX,
    int NY,
    int NZ,
    int rank,
    bool got_north,
    bool got_east,
    bool got_south,
    bool got_west) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //guard
    if(i >=  nx * ny * nz) return;
    //compute indexing for readabilty
    int z = i / (nx * ny);
    int r = (i % (nx * ny));
    int x = r % nx;
    int y = r / nx;
    
    int count = 1;
    float result = data_in[i];
    for(int curr_z = z-1; curr_z<=z+1; curr_z++){
        if(curr_z>=NZ || curr_z<0){
            //all 9 points are zero
            continue;
        }
        int curr_i = i + curr_z * nx * ny;
        //north
        if(got_north){
            if(x == 0){
                //in halo
                result += halo_north[curr_z * nx + x];
            }else{
                result += data_in[curr_i-nx];
            }
            count ++;
            //upper left corner
            if(got_west) {
                if(x == 0 && y == 0){
                    //in halo
                    result += halo_nw[curr_z];
                }else{
                    result += data_in[curr_i-nx-1];
                }
                count++;
            }
        }
        //east
        if(got_east){
            if(y == ny-1){
                //in halo
                result += halo_east[curr_z*ny + y];
            }else{
                result += data_in[curr_i+1];
            }
            count++;
            //upper right corner
            if(got_north) {
                if(x == nx-1 && y == 0){
                    //in halo
                    result += halo_ne[curr_z];
                }else{
                    result += data_in[curr_i-nx+1];
                }
                count++;
            }
        }
        //south
        if(got_south){
            if(y == ny-1){
                //in halo
                result += halo_east[curr_z*nx + x];
            }else{
                result += data_in[curr_i+nx];
            }
            count++;
            //bottom right corner
            if(got_east) {
                if(x == nx-1 && y == ny-1){
                    //in halo
                    result += halo_se[curr_z];
                }else{
                    result += data_in[curr_i+nx+1];
                }
                count++;
            }
        }
        //west
        if(got_west){
            if(x == 0){
                //in halo
                result += halo_east[curr_z*nx + x];
            }else{
                result += data_in[curr_i-1];
            }
            count++;
            //bottom left corner
            if(got_south) {
                if(x == nx-1 && y == 0){
                    //in halo
                    result += halo_sw[curr_z];
                }else{
                    result += data_in[curr_i+nx-1];
                }
                count++;
            }
        }
        
    }
    result /= count;
    data_out[i] = result;
}

bool data_server(int NX, int NY, int NZ, int rank, int size, MPI_Datatype cube_type, MPI_Datatype horizontal_plane_type, MPI_Datatype vertical_plane_type, MPI_Datatype corner_type){
    //init some matrix
    unsigned int N = NX * NY * NZ;
    float *h_A = (float*)malloc(N * sizeof(float));

    // Initialize A. Layout A: [x=0, y=0, z=0], [x=1, y=0, z=0], ..., [x=NX-1, y=0, z=0], [x=0, y=1, z=0], ..., [x=NX-1, y=NY-1, z=0], [x=0, y=0, z=1], [x=1, y=0, z=1], ..., [x=NX-1, y=NY-1, z=NZ-1]
    for (int i = 0; i < N; i++) {
        h_A[i] = (float) i / N;
    }

    //send data
    float *send_addr = h_A;
    for(int i = 1; i<size; i++){
        MPI_Send(send_addr, 1, cube_type, i, 0, MPI_COMM_WORLD);
        if(GOT_NORTH(i)){
            MPI_Send(send_addr - NX, 1, horizontal_plane_type, i, 1, MPI_COMM_WORLD);
            if(GOT_EAST(i)){
                MPI_Send(send_addr + nx - NX, 1, corner_type, i, 2, MPI_COMM_WORLD);
            }
        }
        if(GOT_EAST(i)){
            MPI_Send(send_addr + nx, 1, vertical_plane_type, i, 3, MPI_COMM_WORLD);
            if(GOT_SOUTH(i)){
                MPI_Send(send_addr + nx + ny * NX, 1, corner_type, i, 4, MPI_COMM_WORLD);
            }
        }
        if(GOT_SOUTH(i)){
            MPI_Send(send_addr + ny * NX, 1, horizontal_plane_type, i, 5, MPI_COMM_WORLD);
            if(GOT_WEST(i)){
                MPI_Send(send_addr + nx + ny * NX - 1, 1, corner_type, i, 6, MPI_COMM_WORLD);
            }
        }
        if(GOT_WEST(i)){
            MPI_Send(send_addr - 1, 1, vertical_plane_type, i, 7, MPI_COMM_WORLD);
            if(GOT_NORTH(i)){
                MPI_Send(send_addr - 1 - NX, 1, corner_type, i, 8, MPI_COMM_WORLD);
            }
        }
        send_addr += nx;
        if(!GOT_WEST(i) && i != 0){
            send_addr += (ny - 1) * NX;
        }
        send_addr += nx;
        
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //receive data
    float *recv_addr = h_A;
    for(int i  = 1; i < size; i++){
        MPI_Recv(recv_addr, 1, cube_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(!GOT_WEST(i) && i != 0){
            recv_addr += (ny - 1) * NX;
        }
        recv_addr += nx;
    }

    printf("Some values:\n");
    for(int i = 0; i < N; i += N/10){
            printf("%f ", h_A[i]);
    }
    printf("\n ");
    free(h_A);

    return true;
}

bool compute_process(int NX, int NY, int NZ, int rank, int size, int TSTEPS, MPI_Datatype cube_type, MPI_Datatype horizontal_plane_type, MPI_Datatype vertical_plane_type, MPI_Datatype corner_type){

    float* data_in;
    float* data_out;
    float* halo_north;
    float* halo_east;
    float* halo_south;
    float* halo_west;
    float* halo_ne;
    float* halo_se;
    float* halo_sw;
    float* halo_nw;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount);

    cudaMalloc(&data_in, nx * ny * nz * sizeof(float));
    cudaMalloc(&data_out, nx * ny * nz * sizeof(float));
    MPI_Recv(data_in, nx * ny * nz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(GOT_NORTH(rank)){
        cudaMalloc(&halo_north, nx * ny * sizeof(float));
        MPI_Recv(halo_north, nx * ny, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(GOT_EAST(rank)) {            
            cudaMalloc(&halo_ne, nz * sizeof(float));
            MPI_Recv(halo_ne, nz, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if(GOT_EAST(rank)){
        cudaMalloc(&halo_east, nx * ny * sizeof(float));
        MPI_Recv(halo_east, nx * ny, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(GOT_SOUTH(rank)) {
            cudaMalloc(&halo_se, nz * sizeof(float));
            MPI_Recv(halo_se, nz, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if(GOT_SOUTH(rank)){
        cudaMalloc(&halo_south, nx * ny * sizeof(float));
        MPI_Recv(halo_south, nx * ny, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(GOT_WEST(rank)) {
            cudaMalloc(&halo_sw, nz* sizeof(float));
            MPI_Recv(halo_sw, nz, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if(GOT_WEST(rank)) {
        cudaMalloc(&halo_west, nx * ny * sizeof(float));
        MPI_Recv(halo_west, nx * ny, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(GOT_NORTH(rank)) {
            cudaMalloc(&halo_nw, nz * sizeof(float));
            MPI_Recv(halo_nw, nz, MPI_FLOAT, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    //launch kernel
    int threadsPerBlock = 256;
    int blocks = (nx * ny * nz + threadsPerBlock - 1) / threadsPerBlock;

    for(int i = 0; i<TSTEPS; i++){
        jacobi3D_kernel<<<blocks, threadsPerBlock>>>(
            data_in, 
            data_out, 
            halo_north, 
            halo_east, 
            halo_south, 
            halo_west, 
            halo_ne, 
            halo_se,
            halo_sw, 
            halo_nw, 
            nx, 
            ny, 
            nz, 
            NX, 
            NY, 
            NZ, 
            rank,
            GOT_NORTH(rank),
            GOT_EAST(rank),
            GOT_SOUTH(rank),
            GOT_WEST(rank)
        );

        cudaDeviceSynchronize();
        //exchange halos
        if(GOT_NORTH(rank)){
            //exchange north
            MPI_Sendrecv(data_out, 1, horizontal_plane_type, rank - np_NX, 0, halo_north, nx*ny*nz, MPI_FLOAT, rank - np_NX, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(GOT_EAST(rank)){
                //exchange corner ne
                MPI_Sendrecv(data_out+nx-1, 1, corner_type, rank - np_NX + 1, 1, halo_ne, nz, MPI_FLOAT, rank - np_NX + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if(GOT_EAST(rank)){
            //exchange east
            MPI_Sendrecv(data_out+nx-1, 1, vertical_plane_type, rank+1, 3, halo_east, nx*ny*nz, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(GOT_SOUTH(rank)){
                //exchange corner se
                MPI_Sendrecv(data_out+nx*ny-1, 1, corner_type, rank + np_NX + 1, 4, halo_se, nz, MPI_FLOAT, rank + np_NX + 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if(GOT_SOUTH(rank)){
            //exchange south
            MPI_Sendrecv(data_out+nx*(ny-1), 1, horizontal_plane_type, rank+np_NX, 5, halo_south, nx*ny*nz, MPI_FLOAT, rank + np_NX, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(GOT_WEST(rank)){
                //exchange sw
                MPI_Sendrecv(data_out+nx*(ny-1), 1, corner_type, rank + np_NX - 1, 6, halo_sw, nz, MPI_FLOAT, rank + np_NX - 1, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if(GOT_WEST(rank)){
            //exchange west
            MPI_Sendrecv(data_out, 1, vertical_plane_type, rank-1, 7, halo_west, nx*ny*nz, MPI_FLOAT, rank -1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(GOT_NORTH(rank)){
                //exchange nw
                MPI_Sendrecv(data_out, 1, corner_type, rank - np_NX - 1, 8, halo_nw, nz, MPI_FLOAT, rank - np_NX - 1, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //swap pointers
        float* temp = data_in;
        data_in = data_out;
        data_out = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    //send result
    MPI_Send(data_in, nx * ny * nz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    //free memory
    cudaFree(data_in);
    cudaFree(data_out);
    if(GOT_NORTH(rank)){
        cudaFree(halo_north);
        if(GOT_EAST(rank)) {            
            cudaFree(halo_ne);
        }
    }
    if(GOT_EAST(rank)){
        cudaFree(halo_east);
        if(GOT_SOUTH(rank)) {
            cudaFree(halo_se);
        }
    }
    if(GOT_SOUTH(rank)){
        cudaFree(halo_south);
        if(GOT_WEST(rank)) {
            cudaFree(halo_sw);
        }
    }
    if(GOT_WEST(rank)) {
        cudaFree(halo_west);
        if(GOT_NORTH(rank)) {
            cudaFree(halo_nw);
        }
    }

    return true;
}

bool jacobi3D(int NX, int NY, int NZ, int TSTEPS) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(size - 1 != deviceCount){
        printf("This runs only on deviceCount GPUs with deviceCount + 1 MPI-processes");
        return false;
    }

    np_NX = (int) sqrt(size-1);
    np_NY = np_NX;
    nx = NX / np_NX;
    ny = NY / np_NY;
    nz = NZ;

    if(np_NX * np_NY != size-1){
        printf("Number of processes not supported. Must be square number plus 1.");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }
    if(NX % np_NX != 0|| NY % np_NY != 0){
        printf("Dimeensions must be devisible by sqrt(np-1).");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }

    //define data type
    MPI_Datatype cube_type_0;
    MPI_Type_vector(ny, nx, NX, MPI_FLOAT, &cube_type_0);
    MPI_Type_commit(&cube_type_0);
    MPI_Datatype cube_type;
    MPI_Type_vector(nz, 1, NX * NY, cube_type_0, &cube_type);
    MPI_Type_commit(&cube_type);

    MPI_Datatype horizontal_plane_type;
    MPI_Type_vector(nz, nx, NX * NY, MPI_FLOAT, &horizontal_plane_type);
    MPI_Type_commit(&horizontal_plane_type);

    MPI_Datatype vertical_plane_type_0;
    MPI_Type_vector(ny, 1, NX, MPI_FLOAT, &vertical_plane_type_0);
    MPI_Type_commit(&vertical_plane_type_0);
    MPI_Datatype vertical_plane_type;
    MPI_Type_vector(nz, 1, NX * NY, vertical_plane_type_0, &vertical_plane_type);
    MPI_Type_commit(&vertical_plane_type);

    MPI_Datatype corner_type;
    MPI_Type_vector(nz, 1, NX * NY, MPI_FLOAT, &corner_type);
    MPI_Type_commit(&corner_type);
    
    if(rank == 0){
        data_server(NX, NY, NZ, rank, size, cube_type, horizontal_plane_type, vertical_plane_type, corner_type);
    }else{
        compute_process(NX, NY, NZ, rank, size, TSTEPS,cube_type ,horizontal_plane_type, vertical_plane_type, corner_type);
    }

    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int NX = 128;
    int NY = 128;
    int NZ = 128;
    int TSTEPS = 100;

    jacobi3D(NX, NY, NZ, TSTEPS);

    MPI_Finalize();

    return 0;
}

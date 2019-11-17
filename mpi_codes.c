#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) 
{
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int number;
    
    if (world_rank == 0) 
    {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);        
    } 
    
    else if (world_rank == 1) 
    {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n",number);
    }
    
    MPI_Finalize();
}


/*
#include<mpi.h>
#include<stdio.h>

int main(int argc,char* argv[])
{
    int tags = 42;
    int id,ntasks,source_id,dest_id,i;
    int msg[2];
    MPI_Status status;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

    if(id==0)
    {
        for(i=1;i<ntasks;i++)
        {
            MPI_Recv(msg,2,MPI_INT,MPI_ANY_SOURCE,tags,MPI_COMM_WORLD,&status);
            source_id = status.MPI_SOURCE;
            printf("Received Message %d %d from process %d \n",msg[0],msg[1],source_id);
        }
    }

    else
    {
        msg[0] = id;
        msg[1] = ntasks;
        dest_id = 0;
        MPI_Send(msg,2,MPI_INT,dest_id,tags,MPI_COMM_WORLD);
    }

    MPI_Finalize();

}
*/
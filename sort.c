#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
 
#include <sys/mman.h>
#include <sys/stat.h>
 
#include "mpi.h"
 
#define CHUNK_SIZE_IN_PAGES 100
 
/*
 * System Info:
 * ------------
 *   CPU: 2x E5-2663 (10C)
 *   RAM: 256 GB DDR4 ECC
 *   SSD: 2TB
 * 
 * Previous Benchmarks (Best)
 * -------------------
 *   256 GB Sort: 22 seconds
 *   500 GB Sort: 38 seconds
 *     1 TB Sort: 92 seconds
 */
 
typedef struct {
	uint16_t key[5];
	uint16_t values[45];
} row;

void Consumer(int tid, int fp, unsigned long long size, unsigned long long* offsets, int offsetSize);
void Sort(row* array, row* temp, unsigned long long size);
 
int main (int argc, char* argv[]) {
	int numtasks, rank;
	char* filename = "/mnt/Input/minutesort";
 
	// Initializes all threads to be used for the producer / consumer sort
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
 
	struct stat sb;
	int fp = open(filename, O_RDWR, (mode_t) 0777);
 
	if (fp == -1) {
		perror(filename);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
 
	if (fstat(fp, &sb) == -1) {
		perror("fstat");
		close(fp);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
 
	// Partition file into chunks
	unsigned long long size = sb.st_size;	
	unsigned long long chunkSize = getpagesize() * CHUNK_SIZE_IN_PAGES * 100;
	unsigned long long offset = 0;
	int numberOfPartitions = (int) ceil((double) size / (double) chunkSize);
	int procElements = (int) (numberOfPartitions / numtasks);
	int i;

	unsigned long long* offsets = malloc(numberOfPartitions * sizeof(unsigned long long));
	unsigned long long* subOffsets = malloc(procElements * sizeof(unsigned long long));
	
	for (i = 0; i < numberOfPartitions; ++i) {
		offsets[i] = offset;
		offset += chunkSize;
	}
	
	// Map Reduce into temp output files
	MPI_Scatter(offsets, procElements, MPI_UNSIGNED_LONG_LONG, subOffsets,
				procElements, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	Consumer(rank, fp, size, subOffsets, procElements);
 
	// All nodes call this upon quitting	
	free(offsets);
	free(subOffsets);
	close(fp);
	MPI_Finalize();
}
 
/*
 * Function: Sort
 * --------------
 *   Sorts a list of floats using a radix sort
 *
 *   It splits each element into 16-bit pairs that form the byte array
 *   In order to sort the floats it flips the sign bit of all positive values
 *       and all bits for the negative values
 *
 *   array: uint32_t array representing the bytes of a float array
 *   temp: Temporary storage for the sort to storage semi-sorted values
 *   size: Number of elements in the array
 */
void Sort(row* array, row* temp, unsigned long long size) {
	uint64_t histogram_byte_0[65536] = {0};
	uint64_t histogram_byte_1[65536] = {0};
	uint64_t histogram_byte_2[65536] = {0};
	uint64_t histogram_byte_3[65536] = {0};
	uint64_t histogram_byte_4[65536] = {0};
 
	unsigned long long i;
 
	for (i = 0; i < size; ++i) {
		++histogram_byte_0[array[i].key[0]];
		++histogram_byte_1[array[i].key[1]];
		++histogram_byte_2[array[i].key[2]];
		++histogram_byte_3[array[i].key[3]];
		++histogram_byte_4[array[i].key[4]];
	}
 
	for (i = 1; i < 65536; ++i) {
		histogram_byte_0[i] += histogram_byte_0[i-1];
		histogram_byte_1[i] += histogram_byte_1[i-1];
		histogram_byte_2[i] += histogram_byte_2[i-1];
		histogram_byte_3[i] += histogram_byte_3[i-1];
		histogram_byte_4[i] += histogram_byte_4[i-1];
	}
	
	// Sorts the data by checking the histogram for it's new position
	for (i = size; i-- != 0; ) {
		temp[--histogram_byte_4[array[i].key[4]]] = array[i];
	}
	
	for (i = size; i-- != 0; ) {
		array[--histogram_byte_3[temp[i].key[3]]] = temp[i];
	}
 
	for (i = size; i-- != 0; ) {
		temp[--histogram_byte_2[array[i].key[2]]] = array[i];
	}
 
	for (i = size; i-- != 0; ) {
		array[--histogram_byte_1[temp[i].key[1]]] = temp[i];
	}
 
	for (i = size; i-- != 0; ) {
		temp[--histogram_byte_0[array[i].key[0]]] = array[i];
	}
}
 
/*
 * Function: Consumer
 * ------------------
 *   Sorts chunks of data in the memory mapped file
 *
 *   tid: The threads current id
 *   fp: Input file pointer
 *   size: The size of the input file
 *   offsets: The partition offsets of the file
 *   offsetSize: Number of elements in the offset array
 */
void Consumer(int tid, int fp, unsigned long long size, unsigned long long* offsets, int offsetSize) {
	fprintf(stdout, "Consumer %d:\tStarting consumer\n", tid);
 
	unsigned long long pageSizeInRows = getpagesize() * CHUNK_SIZE_IN_PAGES;
	unsigned long long pageSizeInBytes = getpagesize() * CHUNK_SIZE_IN_PAGES * 100;
	MPI_Status stat;
	double timer;
	int i;
 
	// mmap details
	row *iAddr, *oAddr;
	int fpo;
	char filename[256], buffer[256];
 
 	// Sorts a chunk of the file based on offset retrieved from consumer node
	for (i = 0; i < offsetSize; ++i) {
		unsigned long long input = offsets[i];
		fprintf(stdout, "Consumer %d:\tSorting with offset [%llu]\n", tid, input);
		timer = MPI_Wtime();
 
		if (size - input < pageSizeInBytes) {
			pageSizeInBytes = size - input;
			pageSizeInRows = pageSizeInBytes / 100;
		}
 
		// Read the current memory segments
		iAddr = (row *) mmap(NULL, pageSizeInBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fp, input);
		if (iAddr == MAP_FAILED) {
			sprintf(buffer, "Consumer %d:\tInput mmap for fid [%d] with offset [%llu] and length [%llu]", 
				tid, fp, input, pageSizeInBytes);
			perror(buffer);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
 
		// Creates the new output file to later be merged
		sprintf(filename, "/mnt/Output/SortedResult_%d", stat.MPI_TAG);
		fpo = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0777);
 
		if (fpo == -1) {
			perror(filename);
			munmap(iAddr, pageSizeInBytes);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
 
		if (lseek(fpo, pageSizeInBytes - 1, SEEK_SET) == -1 || write(fpo, "", 1) == -1) {
			sprintf(buffer, "Consumer %d:\tStorage / Set", tid);
			perror(buffer);
			munmap(iAddr, pageSizeInBytes);
			close(fpo);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
 
		oAddr = (row *) mmap(NULL, pageSizeInBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fpo, 0);
		if (oAddr == MAP_FAILED) {
			sprintf(buffer, "Consumer %d:\tResult mmap for file [%s]", tid, filename);
			perror(buffer);
			munmap(iAddr, pageSizeInBytes);
			close(fpo);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
 
		// Sorts the current partition into it's own file
		Sort(iAddr, oAddr, pageSizeInRows);
 
		// Close all the mmaps so far
		munmap(iAddr, pageSizeInBytes);		
		munmap(oAddr, pageSizeInBytes);
		close(fpo);
 
		fprintf(stdout, "Consumer %d:\tFinished sorting at offset [%llu] in [%lf] seconds\n",
				tid, input, MPI_Wtime() - timer);
	}

 	fprintf(stdout, "Consumer %d:\tFinished sorting all partitions\n", tid);
 	MPI_Barrier(MPI_COMM_WORLD);
}
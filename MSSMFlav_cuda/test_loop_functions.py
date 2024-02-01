# test_loop_functions
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

mod = SourceModule("""
  #include <loops.c>
  __global__ void pyfloop(int *i, int *j, int *k, double *z, double *result)
  {
	  int idx = threadIdx.x+blockDim.x*blockIdx.x;
		result[idx] = hyp2f1(i[idx],j[idx],k[idx],z[idx]);
  }
  """)


# i = numpy.random.randint(10, size=4)
# j = numpy.random.randint(10, size=[1,1])
# k = numpy.random.randint(10, size=[1,1])
# i = numpy.random.uniform(0.005, 1.0, size=100000000)
# j = numpy.random.uniform(0.005, 1.0, size=100000000)
# k = numpy.random.uniform(0.005, 1.0, size=100000000)
# i = numpy.array([2,4,2,2,2,2,2,6,2,5,4,3])
# j = numpy.array([2,1,3,3,3,3,3,10,3,3,1,2])
# k = numpy.array([6,5, 5,5,5,5,5,11,5,8,5,5])
# z = numpy.array([-9.84,-6., -1.1, -1. , -1.01, -1.001, -1.0001, -8.41, -1.1, -5.2, -11.5, -9.5])
file = open("test_2f1.txt", "w")

combs = numpy.array([[1,1,2], [1,1,1],[0,2,1],[2,2,2],[1,3,2],[3,1,2],[0,4,2],[3,1,3],[2,1,2], [1,2,2], [1,2,3],[0,3,2]])
a = numpy.zeros(shape=len(combs))
b = numpy.zeros(shape=len(combs))
c = numpy.zeros(shape=len(combs))

for l in range(len(combs)):
	a[l] = combs[l][0]
	b[l] = combs[l][1]
	c[l] = combs[l][2]


z = numpy.random.uniform(-1.15, -0.85, 100)
   
result = numpy.zeros_like(z)
# result_2 = result
# i = numpy.array([1,2,3,4])
# j = numpy.array([1,2,3,4])
# k = numpy.array([1,2,3,4])
# z = numpy.array([1,2,3,4])
# i = i.astype(numpy.int32)
# j = j.astype(numpy.int32)
# k = k.astype(numpy.int32)
# z = z.astype(numpy.float64)


func = mod.get_function("pyfloop")

for l in range(len(combs)):
  i = numpy.repeat(a[l], len(z))
  j = numpy.repeat(b[l], len(z))
  k = numpy.repeat(c[l], len(z))
  # print i, j, k

  i = i.astype(numpy.int32)
  j = j.astype(numpy.int32)
  k = k.astype(numpy.int32)
  z = z.astype(numpy.float64)

  i_gpu = cuda.mem_alloc(i.nbytes)
  j_gpu = cuda.mem_alloc(j.nbytes)
  k_gpu = cuda.mem_alloc(k.nbytes)
  z_gpu = cuda.mem_alloc(z.nbytes)
  result_gpu = cuda.mem_alloc(result.nbytes)


  cuda.memcpy_htod(i_gpu, numpy.repeat(i[l], len(z)))
  cuda.memcpy_htod(j_gpu, numpy.repeat(j[l], len(z)))
  cuda.memcpy_htod(k_gpu, numpy.repeat(k[l], len(z)))
  cuda.memcpy_htod(z_gpu, z)
  cuda.memcpy_htod(result_gpu, result)

  func(i_gpu, j_gpu, k_gpu, z_gpu, result_gpu, 
		      block=(100,1,1), grid=(1,1,1))

  cuda.memcpy_dtoh(result, result_gpu)
  for idx in range(len(result)):
    content = "2F1({0},{1},{2},{3})={4}".format(i[l], j[l], k[l],z[idx], result[idx])
    # print content
    file.write(content)
    file.write("\n")

file.close()
# print result_2
# print z

# vim: fdm=marker ts=2 sw=2 sts=2 sr noet

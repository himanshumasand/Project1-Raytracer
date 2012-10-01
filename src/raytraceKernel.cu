// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  
	ray r;
	glm::vec3 a, b, m, h, v, p;
	
	a = glm::cross(view, up);
	b = glm::cross(a, view);
	m = eye + view;
	h.x = 0.5f; //a * view.length() * fov.x / a.length();
	v.y = 0.5f; 

	p = glm::vec3((float)(m.x + ((((2.0*x)/(resolution.x-1))-1)*h.x)+((((2.0*y)/(resolution.y-1))-1)*v.x)), (float)(m.y + ((((2.0*x)/(resolution.x-1))-1)*h.y)+((((2.0*y)/(resolution.y-1))-1)*v.y)), (float)(m.z + ((((2.0*x)/(resolution.x-1))-1)*h.z)+((((2.0*y)/(resolution.y-1))-1)*v.z)));

	r.origin = eye;
	r.direction = p - eye;
	float mag = (p - eye).length();
	r.direction = r.direction / mag;

	//forDOF
	//float apertureSize, focalLength = 857, numOfRays = 9;
	//glm::vec3 focalPoint;

	//float scdist=(resolution.y/2)/tan(fov.y *22 / 7 / 180);
	//glm::vec3 pixelPos = glm::vec3(x, y, scdist);
	//float pixelDistance = (pixelPos - eye).length();

	//focalPoint = eye + (pixelDistance / (scdist / (scdist + focalLength))) * (pixelPos - eye) / pixelDistance;

	////ray dofRays[9];
	//dofRays[0] = r;
	//dofRays[1].origin = pixelPos + glm::vec3(1,0,0);
	//dofRays[2].origin = pixelPos + glm::vec3(-1,0,0);
	//dofRays[3].origin = pixelPos + glm::vec3(0,1,0);
	//dofRays[4].origin = pixelPos + glm::vec3(0,-1,0);
	//dofRays[5].origin = pixelPos + glm::vec3(1,1,0);
	//dofRays[6].origin = pixelPos + glm::vec3(-1,1,0);
	//dofRays[7].origin = pixelPos + glm::vec3(1,1,0);
	//dofRays[8].origin = pixelPos + glm::vec3(-1,-1,0);

	//for(int i = 0; i < 5; i++)
	//	dofRays[i].direction = focalPoint - dofRays[i].origin;
	//return dofRays;

	return r;
	
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* mats, int numberOfMaterials){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = resolution.x * resolution.y - ( x + (y * resolution.x));
  int currentDepth = 0;
  float dist, distLight;
  float minDist = 99999, minDistLight = 99999;
  int indexOfGeom, indexOfLight;

  const int numOfSamples = 9;
  ray currentRay[numOfSamples], cachedRay[numOfSamples];
  glm::vec3 colorAtDepth[10];
  material materialAtDepth[10];

  glm::vec3 intersectionPoint, normal;

  glm::vec3 lightPosition, lightColor;
  float lightEmittance;
  
  //gets the position of the last defined light source
  for(int i = 0; i < numberOfGeoms; i++)
  {
	  if(mats[geoms[i].materialid].emittance > 0)
	  {
		  lightPosition = geoms[i].translation;
		  indexOfLight = i;
	  }
  }

  lightColor = mats[geoms[indexOfLight].materialid].color;
  lightEmittance = mats[geoms[indexOfLight].materialid].emittance;

  lightPosition = glm::vec3(0,10,0);
  glm::vec3 lightDir;
  //ambient, diffuse and specular factors
  float kAmbient = 0.2f, kDiffuse = 0.5f, kSpecular = 0.3f;

  bool hitDiffuse = false;

  if((x<=resolution.x && y<=resolution.y))
  {
	  currentRay[0] = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[1] = raycastFromCameraKernel(resolution, time, x-0.5f, y, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[2] = raycastFromCameraKernel(resolution, time, x+0.5f, y, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[3] = raycastFromCameraKernel(resolution, time, x, y-0.5f, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[4] = raycastFromCameraKernel(resolution, time, x, y+0.5f, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[5] = raycastFromCameraKernel(resolution, time, x-0.5f, y-0.5f, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[6] = raycastFromCameraKernel(resolution, time, x+0.5f, y-0.5f, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[7] = raycastFromCameraKernel(resolution, time, x-0.5f, y+0.5f, cam.position, cam.view, cam.up, cam.fov);
	  currentRay[8] = raycastFromCameraKernel(resolution, time, x+0.5f, y+0.5f, cam.position, cam.view, cam.up, cam.fov);


	  colors[index] = glm::vec3(0,0,0);
	  
	  while(currentDepth < rayDepth && hitDiffuse == false)		//in the beginning, currentDepth = 0
	  {
		  colorAtDepth[currentDepth] = glm::vec3(0,0,0);
		   
		  for(int ry = 0; ry < numOfSamples; ry++)
		  {
			  for(int i = 0; i < numberOfGeoms; i++)
			  {		  
				  if(geoms[i].type == SPHERE)
					  dist = sphereIntersectionTest( geoms[i] , currentRay[ry], intersectionPoint, normal);
				  else if(geoms[i].type == CUBE)
					  dist = boxIntersectionTest( geoms[i], currentRay[ry], intersectionPoint, normal);

				  if(dist != -1 && dist < minDist)
				  {  
					  minDist = dist;
					  indexOfGeom = i;
				  }
			  }

			  //material curMaterial = mats[geoms[indexOfGeom].materialid];

			  if(minDist == 99999)
			  {
				  colorAtDepth[currentDepth] += glm::vec3(0,0,0);
			  }

			  else
			  {
				  materialAtDepth[currentDepth] = mats[geoms[indexOfGeom].materialid];
				  material curMaterial = mats[geoms[indexOfGeom].materialid];
				  colorAtDepth[currentDepth] += kAmbient * curMaterial.color;
				  
				  if(curMaterial.hasReflective > 0)
				  {
					  normal = glm::normalize(normal);
					  cachedRay[ry].direction = currentRay[ry].direction - 2.0f * (normal * glm::dot(currentRay[ry].direction, normal));
					  cachedRay[ry].direction = glm::normalize(cachedRay[ry].direction);
					  cachedRay[ry].origin = intersectionPoint;
				  }

				  else
					  hitDiffuse = true;

				  ray lightRay;
				  lightDir = lightPosition - intersectionPoint;
				  lightDir = glm::normalize(lightDir);
				  normal = glm::normalize(normal);
				  float factor = glm::dot(normal, lightDir);
				  lightRay.origin = intersectionPoint;
				  lightRay.direction = lightDir;

				  ray reflectedLightRay;
				  reflectedLightRay.direction = lightRay.direction - 2.0f * (normal * glm::dot(lightRay.direction, normal));
				  reflectedLightRay.direction = glm::normalize(reflectedLightRay.direction);
				  reflectedLightRay.origin = intersectionPoint;

				  distLight = -1;
				  minDistLight = 99999;

				  for(int i = 0; i < numberOfGeoms; i++)
				  {
					  if(i != 2 && i != 8)	//hard coded to ignore roof cube intersection during light ray cast
					  {
						  if(geoms[i].type == SPHERE)
							  distLight = sphereIntersectionTest( geoms[i] , lightRay, intersectionPoint, normal);
						  else if(geoms[i].type == CUBE)
							  distLight = boxIntersectionTest( geoms[i], lightRay, intersectionPoint, normal);

						  if(distLight != -1 && distLight < minDistLight)
						  {  
							  minDistLight = distLight;
						  }
					  }
				  }

				  if(minDistLight == 99999)
				  {
					  colorAtDepth[currentDepth] += lightEmittance * lightColor * kDiffuse * factor * curMaterial.color;
					  if(curMaterial.specularExponent > 0)
						  colorAtDepth[currentDepth] += kSpecular * lightEmittance * curMaterial.specularColor * pow(abs(glm::dot(glm::normalize(reflectedLightRay.direction), glm::normalize(currentRay[ry].direction))), curMaterial.specularExponent);
				  }

			  }

			  currentRay[ry] = cachedRay[ry];
		  }//end for loop for rays

		  //currentRay = cachedRay;

		  colorAtDepth[currentDepth] = colorAtDepth[currentDepth] / (float)numOfSamples;
		  currentDepth++;
		  
	  }//end while

	  for(int i = rayDepth - 1; i > 0; i--)
	  {
		  colors[index] += materialAtDepth[i-1].hasReflective * colorAtDepth[i];
	  }
	  colors[index] += colorAtDepth[0];

//	  for(int i = 0; i < rayDepth; i++)
//		  colors[index] += colorAtDepth[i];

	  //average colors if multiple sampling

	  //colors[index] = colors[index] / 5.0f;

	  clamp(colors[index].x, 0.0f, 1.0f);
	  clamp(colors[index].y, 0.0f, 1.0f);
	  clamp(colors[index].z, 0.0f, 1.0f);
   }
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
	newStaticGeom.inverseTranspose = geoms[i].inverseTranspose[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material* matList = new material[numberOfMaterials];
  for(int i = 0; i < numberOfMaterials; i++)
  {
	  material newMaterial;
	  newMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
	  newMaterial.color = materials[i].color;
	  newMaterial.emittance = materials[i].emittance;
	  newMaterial.hasReflective = materials[i].hasReflective;
	  newMaterial.hasRefractive = materials[i].hasRefractive;
	  newMaterial.hasScatter = materials[i].hasScatter;
	  newMaterial.indexOfRefraction = materials[i].indexOfRefraction;
	  newMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	  newMaterial.specularColor = materials[i].specularColor;
	  newMaterial.specularExponent = materials[i].specularExponent;
	  matList[i] = newMaterial;
  }

  material* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamats, matList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamats, numberOfMaterials);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;
  cudaFree( cudamats );
  delete matList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

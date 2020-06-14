/***********************************************************************
*
* Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
* See LICENSE file in the project root for full license information.
*
************************************************************************/

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <iomanip>
#include <exception>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "GTILib.h"
#include "utils.hpp"
#include "net.h"

#if !defined(CV_VERSION_EPOCH) && CV_VERSION_MAJOR == 4
	#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
	#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
	#define CV_CAP_PROP_FORMAT cv::CAP_PROP_FORMAT
	#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
	#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
	#define CV_CAP_PROP_POS_FRAMES cv::CAP_PROP_POS_FRAMES
	#define CV_FONT_HERSHEY_SIMPLEX cv::FONT_HERSHEY_SIMPLEX
	#define cvPoint cv::Point
#endif
//g++ -I /usr/include/ ./VGG_SSD_224_5801.cpp -L /usr/lib/aarch64-linux-gnu/gti -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lfc -lftd3xx -lgticdva -lGTILibrary -limagereader -llabel -lpooling -lsoftmax -lusb-1.0 -l:libncnn.a -lpthread -ldl -lgomp -o ./gti5801_test.out
using namespace std;

#define INPUT_IMAGE_W	224
#define INPUT_IMAGE_H	224
#define CHIPOUT_W 7
#define CHIPOUT_H 7
#define CHIPOUT_C 512
#define CHIPOUT_SCALE 0.9105483870967742

//string sGTIChipModelFile = "/opt/vggmodel/out.model";
//string sGTIPostModelParamFile = "/opt/vggmodel/out.param";
//string sGTIPostModelBinFile = "/opt/vggmodel/out.bin";

string sGTIChipModelFile = "/opt/vggmodel_27/out.model";
string sGTIPostModelParamFile = "/opt/vggmodel_27/out_5801_27.param";
string sGTIPostModelBinFile = "/opt/vggmodel_27/out_5801_27.bin";

#define MAXBUFS 3

class SSDDetection
{
	private:
		GtiModel* pModel;
		ncnn::Net oNet;

		cv::Mat aImage[MAXBUFS];
		float *aChipOutputBuffer[MAXBUFS];
		mutex mutexImgW[MAXBUFS];
		mutex mutexImgR[MAXBUFS];
		mutex mutexChipW[MAXBUFS];
		mutex mutexChipR[MAXBUFS];
		thread *pVideoChipThread = nullptr;
		thread *pVideoSWThread = nullptr;
		bool bTerminate = false;

		TP nStart;
		TP nEnd;

	public:
		SSDDetection()
		{
			pModel = nullptr;
			pVideoChipThread = nullptr;
			pVideoSWThread = nullptr;
		}

		~SSDDetection()
		{
			if (pVideoSWThread)
			{
				pVideoSWThread->join();
				delete pVideoSWThread;
			}
			if (pVideoChipThread)
			{
				pVideoChipThread->join();
				delete pVideoChipThread;
			}			
			for (int nI=0; nI<MAXBUFS; nI++)
			{
				if (aChipOutputBuffer[nI])
					delete [] aChipOutputBuffer[nI];
			}			
			if (pModel)
			{
				GtiDestroyModel(pModel);
				pModel = nullptr;
			}
		}
		
		void init()
		{
			printf("GTI init ...\n");
			printf("GtiCreateModel %s before ...\n",sGTIChipModelFile.c_str());
			pModel = GtiCreateModel(sGTIChipModelFile.c_str());
			printf("GtiCreateModel %s success ...\n",sGTIChipModelFile.c_str());
			if (pModel == nullptr)
			{
				throw runtime_error("Failed to load model.");
			}
#if defined(__arm__) || defined(__aarch64__)	
			oNet.opt.use_winograd_convolution = false;	
#endif
			oNet.load_param(sGTIPostModelParamFile.c_str());
			printf("load_param done\n");
			oNet.load_model(sGTIPostModelBinFile.c_str());
			printf("load_model done\n");

			for (int nI=0; nI<MAXBUFS; nI++)
			{
				aChipOutputBuffer[nI] = new float[CHIPOUT_C*CHIPOUT_H*CHIPOUT_W];
				mutexImgR[nI].lock();
				mutexChipR[nI].lock();
			}
		}

		void forwardGTI(cv::Mat& rImg, float* pOutputBuffer)
		{
			printf("forwardGTI ...\n");
			// resize image to model required resolution
			cv::Mat oImgResized;
			cv::resize(rImg, oImgResized, cv::Size(INPUT_IMAGE_H, INPUT_IMAGE_W));
			// fill input buffer in CHW format
			int nInputLength = INPUT_IMAGE_W * INPUT_IMAGE_H * 3;
			uint8_t* pInputBuffer = new uint8_t[nInputLength];
			int nIndex = 0;
			for (int nC=0; nC<3; nC++) 
			{
				for (int nH=0; nH<oImgResized.rows; nH++) 
				{
					for (int nW=0; nW<oImgResized.cols; nW++)
					{
						uint8_t nValue = static_cast<uint8_t>(oImgResized.at<cv::Vec3b>(nH, nW)[nC]);
						pInputBuffer[nIndex++] = uint8_t(nValue * (31.0/255));
					}
				}
			}
			// make input tensor	
			GtiTensor oTensor;
			oTensor.height = INPUT_IMAGE_H;
			oTensor.width = INPUT_IMAGE_W;
   			oTensor.depth = 3;
			oTensor.buffer = pInputBuffer;
			oTensor.size = nInputLength; 
			// run inference on chip
			GtiTensor *pTensorOut=GtiEvaluate(pModel, &oTensor);
			if(pTensorOut == 0)
			{
				delete [] pInputBuffer;
				throw runtime_error("GTI ERROR");
			}
			char* pChipOutBuffer = (char*)pTensorOut->buffer;

			// update chip output with output scale
			for (int nI=0; nI<pTensorOut->size; nI++)
			{
				pOutputBuffer[nI] = (float)pChipOutBuffer[nI] * CHIPOUT_SCALE / 8;

			}
			
			delete [] pInputBuffer;			
		}

		void forwardSW(cv::Mat& rImg, float* pChipOutputBuffer)
		{
			ncnn::Extractor oExt = oNet.create_extractor();	
			// the rest of the model has 2 inputs
			// chip output layer "scale"
			ncnn::Mat inputData(CHIPOUT_W, CHIPOUT_H, CHIPOUT_C, sizeof(float));
			int nChipOutChannelSize = CHIPOUT_W * CHIPOUT_H;
			for(int nC=0; nC<inputData.c; nC++)
			{
				memcpy((void*)(inputData.channel(nC)), pChipOutputBuffer + nC * nChipOutChannelSize, nChipOutChannelSize * sizeof(float));
			}
			oExt.input("scale", inputData);
			// image data layer "data"
			ncnn::Mat imageData(INPUT_IMAGE_W, INPUT_IMAGE_H, 3, sizeof(float));
			oExt.input("data", imageData);
			// output layer detection output
			ncnn::Mat outData;
			oExt.extract("detection_out",outData);

			vector<vector<float> > vvObject;
			for (int nI=0; nI<outData.h; nI++)
			{
				const float* pValues = outData.row(nI);

				vector<float> vObject(6);
				vObject[0] = pValues[0];
				vObject[1] = pValues[1];
				vObject[2] = pValues[2] * rImg.cols;
				vObject[3] = pValues[3] * rImg.rows;
				vObject[4] = pValues[4] * rImg.cols;
				vObject[5] = pValues[5] * rImg.rows;
				//cerr << vObject[0] << " " << vObject[1] << " (" << vObject[2] << " " << vObject[3] << " " << vObject[4] << " " << vObject[5] << ")" << endl;
				vvObject.push_back(vObject);
			}
			draw(rImg, vvObject);
		}

		void draw(cv::Mat& rImg, vector<vector<float> >& rvvObject)
		{
			/**const char* apClassNames[] = {"background",
				"bicycle", "car", "bus", "person", "motobike"};*/
			const char* apClassNames[] = {"background","face"};

			for (size_t nI=0; nI<rvvObject.size(); nI++)
			{
				cv::Rect oRect(cv::Point(rvvObject[nI][2], rvvObject[nI][3]), cv::Point(rvvObject[nI][4], rvvObject[nI][5]));
				cv::rectangle(rImg, oRect, cv::Scalar(255, 0, 0));

				stringstream ss;
				ss << apClassNames[int(rvvObject[nI][0])] << " " << fixed << setprecision(4) << rvvObject[nI][1];

				cv::putText(rImg, ss.str(), cv::Point(rvvObject[nI][2], rvvObject[nI][3]),
							cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
			}
			//cv::imwrite("/opt/rockchip-mpp/outfile/out_dect_5801.jpg",rImg);
			cv::imshow("SSDDetection", rImg);
		}

		void processImage(string sImageName)
		{
			long start, end;
    		start = clock();
			cv::Mat oImg = cv::imread(sImageName, CV_LOAD_IMAGE_COLOR);
			if (oImg.empty())
			{
				stringstream ss;
				ss << "failed to read image " << sImageName;
				throw runtime_error(ss.str());
			}
			forwardGTI(oImg, aChipOutputBuffer[0]);
			forwardSW(oImg, aChipOutputBuffer[0]);
			cv::waitKey(0);
			end = clock();
    		cout << "dect img:" << (double)(end-start)/CLOCKS_PER_SEC*1000 << "ms" << endl;
		}

		void processVideo(string sVideoName)
		{
			pVideoChipThread = new thread(&SSDDetection::processVideoPipeChip, this);
			pVideoSWThread = new thread(&SSDDetection::processVideoPipeSW, this);
	
			cv::VideoCapture oCap;
			
			if (!sVideoName.empty())
			{
				bool bRet = oCap.open(sVideoName);
				cout << "Open Video file " << sVideoName << " status: " << bRet << endl;
				if (!oCap.isOpened())
				{
					oCap.release();
					stringstream ss;
					ss << "Video file " << sVideoName << " is not opened!" << endl;
					throw runtime_error(ss.str());
				}
			}
			else
			{
				oCap.set(CV_CAP_PROP_FORMAT, CV_8UC3);
				oCap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
				oCap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
				if (!oCap.open(0))
				{
					throw runtime_error("Failed to open camera.");
				}
			}

			nStart = getTime();
			int nBufIdx = 0;
			while (1)
			{
				mutexImgW[nBufIdx].lock();
				if (bTerminate)
					break;
				if (!oCap.read(aImage[nBufIdx]))
				{
					// reset if video playing ended
					oCap.set(CV_CAP_PROP_POS_FRAMES, 0);
					oCap.read(aImage[nBufIdx]);
				}
				mutexImgR[nBufIdx].unlock();
				nBufIdx = (nBufIdx+1) % MAXBUFS;
			}
		}

		void processVideoPipeChip()
		{
			int nBufIdx = 0;
			while(1)
			{
				mutexImgR[nBufIdx].lock();
				if (bTerminate)
					break;
				mutexChipW[nBufIdx].lock();
				if (bTerminate)
					break;
				forwardGTI(aImage[nBufIdx], aChipOutputBuffer[nBufIdx]);
				mutexChipR[nBufIdx].unlock();
				nBufIdx = (nBufIdx+1) % MAXBUFS;
			}
		}

		void processVideoPipeSW()
		{
			int nCount = 0;
			float nFPS = 0;
			int nBufIdx = 0;
			bool bPauseFlag = false;
			while (1)
			{
				if (!bPauseFlag)
				{	
					mutexChipR[nBufIdx].lock();
					forwardSW(aImage[nBufIdx], aChipOutputBuffer[nBufIdx]);
					mutexImgW[nBufIdx].unlock();
					mutexChipW[nBufIdx].unlock();
					nBufIdx = (nBufIdx+1) % MAXBUFS;
					 
					if ((nCount+1) % 20 == 0)
					{
						nEnd = getTime();
						// The inferercing time is afftected by parallel running of image decoding, processing, drawing and showing.
						// On Raspberry Pi 3 , we estimated the time is 16000us per frame. We excuded the time in FPS calculation.  
#ifdef PI3
						nFPS = getFPS(nStart, nEnd, 20, 16000);
#else
						nFPS = getFPS(nStart, nEnd, 20);
#endif
						cerr << "fps " << nFPS << endl;
						nStart = getTime();
					}
					nCount++;

					stringstream ss;
					ss << "fps " << fixed << setprecision(2) << nFPS;
					cv::putText(aImage[nBufIdx], ss.str(), cv::Point(10, 20),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));					
				}
				int nKey = cv::waitKey(1) & 0xff;
				if (nKey == 'q' || nKey == 27 || nKey == 'n')
				{
					bTerminate = true;
					for(int nX=0; nX<MAXBUFS; nX++)
					{
						mutexChipW[nX].unlock();
						mutexImgR[nX].unlock();
						mutexImgW[nX].unlock();
					}
					break;
				}
				else if (nKey == ' ')
				{
					bPauseFlag = !bPauseFlag;
				}				
			}
		}
};

int main(int nArgc, char** ppArgv)
{
	int nMode;
	string sFileName;
	if (nArgc == 2)
	{
		sFileName = string(ppArgv[1]);
		if(sFileName.compare("camera") == 0)
		{
			nMode = 2;
		}
		else
		{
			cv::Mat oImg = cv::imread(ppArgv[1], CV_LOAD_IMAGE_COLOR);
			if (oImg.empty())
			{
				cv::VideoCapture oCap(sFileName);
				if (!oCap.isOpened())
				{
					cerr << "failed to read input file " << ppArgv[1] << endl;
					return -1;
				}
				else
				{
					cv::Mat oTemp;
					if (oCap.read(oTemp))
					{
						nMode = 1;
						oCap.release();
					}
				}
			}
			else
			{
				nMode = 0;
			}
		}
	}
	else
	{
		cerr << "Usage: " << ppArgv[0] << " [ImageFile|VideoFile|\"camera\"]" << endl;
		return -1;
	}

	try
	{
		SSDDetection oDet;
		oDet.init();

		if (nMode == 0)
			oDet.processImage(sFileName);
		else if (nMode == 1)
			oDet.processVideo(sFileName);
		else
			oDet.processVideo("");
	}
	catch (exception & rE)
	{
		cerr << rE.what() << endl;
	}
	return 0;
}

#include <stdio.h>
#include <tchar.h>
#include <Windows.h>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include <algorithm>    // std::random_shuffle
#include <iostream>
#include <vector>

#include "opencv2\opencv.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;


void readTrainData(vector<string> &names)
{
	_WIN32_FIND_DATAA FindFileData;
	HANDLE hf;
	hf=FindFirstFileA("..\\data\\train\\*.jpg", &FindFileData);
	if (hf!=INVALID_HANDLE_VALUE)
	{
		do
		{
			string fn =  FindFileData.cFileName;
			names.push_back(fn);
		}
		while (FindNextFileA(hf,&FindFileData)!=0);
		FindClose(hf);
	}



	 std::srand ( unsigned ( std::time(0) ) );
	random_shuffle(names.begin(), names.end());
}

string getPrefix(const string &s)
{
	string res;
	for (int i = 0; i < s.size(); i++)
	{
		if (s[i] == '.')
			break;
		res += s[i];
	}
	return res;
}

int main()
{
	system("pause");

	cout << "Process..." << endl;
	vector<string> names;
	readTrainData(names);

	vector<Mat> images;
	vector<int> labels;
	Mat img;

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(1, 8, 8, 8, 60.0);//createEigenFaceRecognizer(10, 1000);

	vector<string> v[1000];
	int classInd = 0;

	v[0].push_back(names[0]);
	img = imread("..\\data\\train\\"+names[0], CV_LOAD_IMAGE_GRAYSCALE);
	images.push_back(img);
	labels.push_back(0);	
	model->train(images, labels);
	//model->update(
	for (int i = 1; i < names.size(); i++)
	{
		images.clear();
		labels.clear();
		img = imread("..\\data\\train\\"+names[i], CV_LOAD_IMAGE_GRAYSCALE);
		int res  = model->predict(img);
		if (res == -1)
		{
			res = ++classInd;
		}

		if (i * 100 / names.size() % 10 == 0)
			cout << i * 100 / names.size() << "%" << endl;

		images.push_back(img);
		labels.push_back(res);	
		v[res].push_back(names[i]);

		model->update(images, labels);
	}
	cout << "Done" << endl;
	//lets calc error 
	//if you wanna know error you need to rename faces of one person with same prefix
	int numOfClasses = v[classInd].size() != 0 ? classInd + 1 : classInd;

	for (int i = 0; i < numOfClasses; i++)
	{
		cout << i << endl;
		string fn = "..\\data\\result\\" + to_string(_Longlong(i));
		CreateDirectoryA(fn.c_str(), NULL);
		//CopyFile(L"train\\astefa.8.jpg", L"result\\lol.jpg", false);
		for (int j = 0; j < v[i].size(); j++)
		{
			string from = "..\\data\\train\\" + v[i][j];
			string to = fn + "\\" + v[i][j];
			CopyFileA(from.c_str(), to.c_str(), false);
			cout << v[i][j] << endl;
		}
	}

	int error = 0;
	string prefixes[1000];
	
	for (int i = 0; i < numOfClasses; i++)
	{
		prefixes[i] = getPrefix(v[i][0]);
		for (int j = 0; j < i; j++)
			if (prefixes[j] == prefixes[i])
			{
				prefixes[i] += "ERROR";
				error++;
			}
		for (int j = 1; j < v[i].size(); j++)
		{
			string s = getPrefix(v[i][j]);
			if ( s != prefixes[i])
				error++;
		}
	}

	cout << "Error images: " << error << " of " << names.size() << endl;
	cout << "Precision: " << 1.0f - float(error) / float(names.size()) << endl;

	system("pause");
	return 0;
}
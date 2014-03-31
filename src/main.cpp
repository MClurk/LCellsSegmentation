/*  Zang Hui
 *  huezann@gmail.com
 *  superpixel pre-processing + GraphCut image segmentation
 *  2013 - 2014
 */

#include <ctime>
#include <cstdio>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "lcells_graphcut_segmentation.h"
using namespace cv;

struct MouseParam {
	Mat src_image, paint_image, seed_image;
	Point pre_pt;
	Color color, gray;
};  //to make the param like a global variable

static void onMouse( int event, int x, int y, int flags, void* param)
{
	MouseParam* ptr_local_mouse_param = (MouseParam*)param;
	const Mat &src_image = ptr_local_mouse_param->src_image;
	if (src_image.empty())
		return;
	Mat &paint_image = ptr_local_mouse_param->paint_image;
	Mat &seed_image = ptr_local_mouse_param->seed_image;
	Point &pre_pt = ptr_local_mouse_param->pre_pt;
	Color &color = ptr_local_mouse_param->color;
	Color &gray = ptr_local_mouse_param->gray;
	  //draw to label the pixels
    if (EVENT_LBUTTONDOWN != event &&
    	EVENT_RBUTTONDOWN != event &&
    	flags != EVENT_FLAG_LBUTTON &&
    	flags != EVENT_FLAG_RBUTTON) {}  //do nothing
	else if(EVENT_LBUTTONDOWN == event)
	{
		pre_pt = Point(x,y);
		color = Color(0, 0, 255);
		gray  = Color(255, 255, 255);
	}  //before left button draw, change the color of stroke
	else if(EVENT_RBUTTONDOWN == event)
	{
		pre_pt = Point(x,y);
		color = Color(255, 0, 0);
		gray  = Color(128, 128, 128);
	}  //before right button draw, change the color of stroke
	else if( event == CV_EVENT_MOUSEMOVE )
	{
		Point pt(x,y);
		line( seed_image, pre_pt, pt, gray, 5);
		line( paint_image, pre_pt, pt, color, 5);
		pre_pt = pt;
		imshow("paint on it!", paint_image);
	}  //when drawings
}

static void mainHelp() {
	printf( "Hot keys help: \n"
		"\tESC - exit \n"
		"\tr - redraw \n"
		"\tw - run the exe\n"
		"\tleft click to mark object versus right click to mark background\n" );
}

int main(int argc, char** argv)
{
	if (argc != 3)
		return -1;  //path of image, path of super pixel labels.txt, [weight between two energy cost]
	MouseParam mouse_param;
	mouse_param.src_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	const Mat &src_image = mouse_param.src_image;
	mouse_param.paint_image = src_image.clone();
	mouse_param.seed_image.create(src_image.rows, src_image.cols, CV_8UC1);
	mouse_param.seed_image = Scalar::all(0);

	mainHelp();
	LcellsGraphcutSegmentation object_segment(src_image, argv[2], 0.5/*lambda*/);
	imshow("paint on it!", mouse_param.paint_image);
	setMouseCallback("paint on it!", onMouse, (void*)(&mouse_param));
	bool esc_or_not = true;
	while(esc_or_not)
	{
		switch (waitKey(0)) {
		  case 27:
			  esc_or_not = false;
			  break;
		  case 'r':
			  mouse_param.seed_image = Color::all(0);
			  src_image.copyTo(mouse_param.paint_image);
			  imshow("paint on it!", mouse_param.paint_image );
			  break;
		  case 'w':
			  //clock_t nTime = clock();
			  object_segment.runSegment( mouse_param.seed_image );
			  //nTime = clock() - nTime;
			  //cout<<"Running Time: "<< nTime / CLOCKS_PER_SEC << "s" << endl;
			  break;
		  default:
			  printf("wrong key! read the help and try again!");
		}  //end of switch
	}
	waitKey(0);
	return 0;
}

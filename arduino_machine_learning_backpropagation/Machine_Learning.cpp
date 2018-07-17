//###########################################################################
//####    Demo KNN Classification of Iris Data Set by Fisher 1936        ####
//####             Training Data Set / Test Dat Set                      ####
//####        Make what you want with this sourcecode :-)                ####
//####            Thinking about: shit in > shit out                     ####
//###########################################################################

//This code you can use for example by Visual Studio 2015 as c++ console.application

#include "stdafx.h"
#include <iostream>
#include <string>

using namespace std;

#define num_of_inputs 5   // neurons
#define num_of_hiddens 5  // neurons
#define num_of_outputs 4  // neurons
#define num_of_weights 6  // 1 x neuron = input, input, input, input, input, bias
#define num_of_neurons 14 // total neurons
#define num_of_layers 3   // output, hidden 2, hidden 1

float inputs[num_of_inputs] = {};   // input neuron data
float outputs[num_of_outputs] = {}; // output neuron data
string objects[num_of_outputs] = { "Iris-setosa", "Iris-versicolor", "Iris-virginica", "" }; // alias of the 4 output neurons
//--------------------------------------------------------------------
float weights[num_of_neurons][num_of_weights] = { //contains all weights and bias values

	// 1 x neuron = input, input, input, input, input, output

};
//--------------------------------------------------------------------
int max_gradient_x; // x position in gradients table
int max_gradient_y; // y position in gradients table

float gradients[num_of_neurons][num_of_weights] = { // values to search the highest improvement of error
};
//--------------------------------------------------------------------
const int num_of_training_data_set = 21; //length of table
float training_data_set[num_of_training_data_set][num_of_inputs + num_of_outputs] = { //input and output data to learn
 //5x input                 |4x output
 // Training Data:
	{ 5.1, 3.5, 1.4, 0.2, 0, 1, 0, 0, 0 },  // Iris-setosa          IRIS Data from Fisher 1936
	{ 4.9, 3.0, 1.4, 0.2, 0, 1, 0, 0, 0 },  //                      https://en.wikipedia.org/wiki/Iris_flower_data_set
	{ 4.7, 3.2, 1.3, 0.2, 0, 1, 0, 0, 0 },
	{ 4.6, 3.1, 1.5, 0.2, 0, 1, 0, 0, 0 },
	{ 4.6, 3.2, 1.4, 0.2, 0, 1, 0, 0, 0 },
	{ 5.3, 3.7, 1.5, 0.2, 0, 1, 0, 0, 0 },
	{ 5.0, 3.3, 1.4, 0.2, 0, 1, 0, 0, 0 },

	{ 7.0, 3.2, 4.7, 1.4, 0, 0, 1, 0, 0 }, // Iris-versicolor
	{ 6.4, 3.2, 4.5, 1.5, 0, 0, 1, 0, 0 },
	{ 6.9, 3.1, 4.9, 1.5, 0, 0, 1, 0, 0 },
	{ 5.5, 2.3, 4.0, 1.3, 0, 0, 1, 0, 0 },
	{ 6.2, 2.9, 4.3, 1.3, 0, 0, 1, 0, 0 },
	{ 5.1, 2.5, 3.0, 1.1, 0, 0, 1, 0, 0 },
	{ 5.7, 2.8, 4.1, 1.3, 0, 0, 1, 0, 0 },

	{ 6.3, 3.3, 6.0, 2.5, 0, 0, 0, 1, 0 }, // Iris-virginica
	{ 5.8, 2.7, 5.1, 1.9, 0, 0, 0, 1, 0 },
	{ 7.1, 3.0, 5.9, 2.1, 0, 0, 0, 1, 0 },
	{ 6.3, 2.9, 5.6, 1.8, 0, 0, 0, 1, 0 },
	{ 6.5, 3.0, 5.2, 2.0, 0, 0, 0, 1, 0 },
	{ 6.2, 3.4, 5.4, 2.3, 0, 0, 0, 1, 0 },
	{ 5.9, 3.0, 5.1, 1.8, 0, 0, 0, 1, 0 },
};
//--------------------------------------------------------------------
const int num_of_test_data_set = 9; // length of table, count of object
float test_data_set[num_of_test_data_set][num_of_inputs] = { // input data to predict
//5x input
	{ 4.8, 3.4, 1.6, 0.2, 0 }, // Iris-setosa
	{ 4.8, 3.0, 1.4, 0.1, 0 },
	{ 4.3, 3.0, 1.1, 0.1, 0 },

	{ 5.6, 2.7, 4.2, 1.3, 0 }, // Iris-versicolor
	{ 5.5, 2.5, 4.0, 1.3, 0 },
	{ 5.6, 3.0, 4.1, 1.3, 0 },

	{ 7.4, 2.8, 6.1, 1.9, 0 }, // Iris-virginica
	{ 7.9, 3.8, 6.4, 2.0, 0 },
	{ 6.4, 2.8, 5.6, 2.2, 0 },
};
//######################################################################################################
float learn_rate = 0; // dynamic calculation
int iterations_counter = 0;
int maximum_iterations = 100000;
float accepted_error = 0.01;
int learn_extra_rounds = 3; //if the software cant find a global minimum > init the weights new.
float threshold = 0.85; //for a good prediction
unsigned long time;
int use_init_weights = 0;// 0 = init by random weights, 1 = init by learned weights
//######################################################################################################

float calc_neuron(float input_0, float weight_0, float input_1, float weight_1, float input_2, float weight_2, float input_3, float weight_3, float input_4, float weight_4, float weight_output) { // this is one neuron

	float output = weight_output + (input_0 * weight_0) + (input_1 * weight_1) + (input_2 * weight_2) + (input_3 * weight_3) + (input_4 * weight_4);
	return output;
}
//-----------------------------------------------------------------------------------------------------------------
float sigmoid(float x) { //sigmoid funktion

	float e = 2.71828; float gain = 2;
	float result;
	result = 1 / (1 + (pow(e, (gain * x * -1))));
	return result;
}
//-----------------------------------------------------------------------------------------------------------------
void calc_neuron_net(float x0, float x1, float x2, float x3, float x4) { // calculate all output values by input values

    //######first hidden layer##########################
	float y0 = calc_neuron(x0, weights[0][0], x1, weights[0][1], x2, weights[0][2], x3, weights[0][3], x4, weights[0][4], weights[0][5]);
	y0 = sigmoid(y0);
	//cout << "Y0=" << y0 << endl; 
	//-------------------------------------------------
	float y1 = calc_neuron(x0, weights[1][0], x1, weights[1][1], x2, weights[1][2], x3, weights[1][3], x4, weights[1][4], weights[1][5]);
	y1 = sigmoid(y1);
	//cout << "Y1=" << y1 << endl; 
	//-------------------------------------------------
	float y2 = calc_neuron(x0, weights[2][0], x1, weights[2][1], x2, weights[2][2], x3, weights[2][3], x4, weights[2][4], weights[2][5]);
	y2 = sigmoid(y2);
	//cout << "Y2=" << y2 << endl; 
	//-------------------------------------------------
	float y3 = calc_neuron(x0, weights[3][0], x1, weights[3][1], x2, weights[3][2], x3, weights[3][3], x4, weights[3][4], weights[3][5]);
	y3 = sigmoid(y3);
	//cout << "Y3=" << y3 << endl; 
	//-------------------------------------------------
	float y4 = calc_neuron(x0, weights[4][0], x1, weights[4][1], x2, weights[4][2], x3, weights[4][3], x4, weights[4][4], weights[4][5]);
	y4 = sigmoid(y4);
	//cout << "Y4=" << y4 << endl; 

	//######second hidden layer##########################
	float y00 = calc_neuron(y0, weights[5][0], y1, weights[5][1], y2, weights[5][2], y3, weights[5][3], y4, weights[5][4], weights[5][5]);
	y00 = sigmoid(y00);
	//cout << "y00=" << y00 << endl; 
	//-------------------------------------------------
	float y01 = calc_neuron(y0, weights[6][0], y1, weights[6][1], y2, weights[6][2], y3, weights[6][3], y4, weights[6][4], weights[6][5]);
	y01 = sigmoid(y01);
	//cout << "y01=" << y01 << endl; 
	//-------------------------------------------------
	float y02 = calc_neuron(y0, weights[7][0], y1, weights[7][1], y2, weights[7][2], y3, weights[7][3], y4, weights[7][4], weights[7][5]);
	y02 = sigmoid(y02);
	//cout << "y02=" << y02 << endl; 
	//-------------------------------------------------
	float y03 = calc_neuron(y0, weights[8][0], y1, weights[8][1], y2, weights[8][2], y3, weights[8][3], y4, weights[8][4], weights[8][5]);
	y03 = sigmoid(y03);
	//cout << "y03=" << y03 << endl; 
	//-------------------------------------------------
	float y04 = calc_neuron(y0, weights[9][0], y1, weights[9][1], y2, weights[9][2], y3, weights[9][3], y4, weights[9][4], weights[9][5]);
	y04 = sigmoid(y04);
	//cout << "y04=" << y04 << endl; 

	//######output layer#################################
	float out0 = calc_neuron(y00, weights[10][0], y01, weights[10][1], y02, weights[10][2], y03, weights[10][3], y04, weights[10][4], weights[10][5]);
	out0 = sigmoid(out0);
	outputs[0] = out0;
	//cout << "Out0=" << out0 << endl; 
	//-------------------------------------------------
	float out1 = calc_neuron(y00, weights[11][0], y01, weights[11][1], y02, weights[11][2], y03, weights[11][3], y04, weights[11][4], weights[11][5]);
	out1 = sigmoid(out1);
	outputs[1] = out1;
	//cout << "Out1=" << out1 << endl; 
	//-------------------------------------------------
	float out2 = calc_neuron(y00, weights[12][0], y01, weights[12][1], y02, weights[12][2], y03, weights[12][3], y04, weights[12][4], weights[12][5]);
	out2 = sigmoid(out2);
	outputs[2] = out2;
	//cout << "Out2=" << out2 << endl; 
	//-------------------------------------------------
	float out3 = calc_neuron(y00, weights[13][0], y01, weights[13][1], y02, weights[13][2], y03, weights[13][3], y04, weights[13][4], weights[13][5]);
	out3 = sigmoid(out3);
	outputs[3] = out3;
	//cout << "Out3=" << out3 << endl; 
}
//-----------------------------------------------------------------------------------------------------------------
float calc_error() { // find the whole error of neuron network by input the training data

	float total_error = 0;
	float error_0;
	float error_1;
	float error_2;
	float error_3;

	for (int i = 0; i < num_of_training_data_set; i++) {

		calc_neuron_net(training_data_set[i][0], training_data_set[i][1], training_data_set[i][2], training_data_set[i][3], training_data_set[i][4]);

		error_0 = pow((training_data_set[i][5] - outputs[0]), 2);
		error_1 = pow((training_data_set[i][6] - outputs[1]), 2);
		error_2 = pow((training_data_set[i][7] - outputs[2]), 2);
		error_3 = pow((training_data_set[i][8] - outputs[3]), 2);

		total_error += error_0;
		total_error += error_1;
		total_error += error_2;
		total_error += error_3;
	}

	//cout << "Total Error=" << total_error << endl; 
	total_error *= 0.5;
	return total_error;
}
//-----------------------------------------------------------------------------------------------------------------
void calc_max_gradient(int from_neuron, int to_neuron) { // search max. gradient of one layer

	float total_error = calc_error();
	float max_steigung = 0;

	for (int y = from_neuron; y <= to_neuron; y++) {
		for (int x = 0; x < num_of_weights; x++) {

			weights[y][x] += learn_rate;

			float new_error = calc_error();
			float diff = total_error - new_error;

			gradients[y][x] = diff;

			//cout << "Gradient=" << diff << " Position=[" << y + "][" << x << "]" << " Total Error=" << total_error << endl; 

			weights[y][x] -= learn_rate;

			if (fabs(diff) > max_steigung) {
				max_steigung = fabs(diff);
				max_gradient_y = y;
				max_gradient_x = x;
			}
		}
	}
}
//-----------------------------------------------------------------------------------------------------------------
string max_classification() {

	float result = 0;
	string alias;

	for (int i = 0; i < num_of_outputs; i++) {
		if (outputs[i] > result) {
			result = outputs[i];
			alias = objects[i];
		}
	}

	if (result < threshold) alias = "n/a";
	return alias;
}
//-----------------------------------------------------------------------------------------------------------------
void runtime() {


}
//-----------------------------------------------------------------------------------------------------------------
void show_weights() { //you can copy the serial output of weights into the sourcecode

	cout << "------------------------" << endl;
	cout << "Show Weights:" << endl;
	for (int y = 0; y < num_of_neurons; y++) {
		for (int x = 0; x < num_of_weights; x++) {
			cout << "weights[" << y << "][" << x << "] = " << weights[y][x] << ";" << endl;
		}
	}
}
//-----------------------------------------------------------------------------------------------------------------
void start_predict() {

	cout << "------------------------" << endl;
	cout << "Start Predict:" << endl;

	cout << "Training Data:" << endl;

	for (int i = 0; i < num_of_training_data_set; i++) {

		calc_neuron_net(training_data_set[i][0], training_data_set[i][1], training_data_set[i][2], training_data_set[i][3], training_data_set[i][4]); //input training data

		cout << " Out 0 = " << outputs[0];
		cout << " / Out 1 = " << outputs[1];
		cout << " / Out 2 = " << outputs[2];
		cout << " / Out 3 = " << outputs[3];

		cout << " = " + max_classification() << endl;

	}

	cout << "Test Data:" << endl;

	for (int i = 0; i < num_of_test_data_set; i++) {

		calc_neuron_net(test_data_set[i][0], test_data_set[i][1], test_data_set[i][2], test_data_set[i][3], test_data_set[i][4]); //input test data

		cout << " Out 0 = " << outputs[0];
		cout << " / Out 1 = " << outputs[1];
		cout << " / Out 2 = " << outputs[2];
		cout << " / Out 3 = " << outputs[3];

		cout << " = " + max_classification() << endl;

	}

	cout << "Stop  Predict" << endl;
	cout << "------------------------" << endl;
}
//-----------------------------------------------------------------------------------------------------------------
void start_learning() {

	cout << "Start Learning" << endl;
	cout << "------------------------" << endl;
	float total_error;
	int used_layer = 0;

	do { //search for global minimum

		 //Backpropagation: training the layers from back to front
		if (used_layer == 0)  calc_max_gradient(10, 13);     // number of output layer  = from 10 to 13 neuron
		if (used_layer == 1)  calc_max_gradient(5, 9);       // number of hidden layer 2= from 5 to 9 neuron
		if (used_layer == 2)  calc_max_gradient(0, 4);       // number of hidden layer 1= from 0 to 4 neuron

		if (gradients[max_gradient_y][max_gradient_x] > 0) weights[max_gradient_y][max_gradient_x] += learn_rate;
		if (gradients[max_gradient_y][max_gradient_x] < 0) weights[max_gradient_y][max_gradient_x] -= learn_rate;

		total_error = calc_error();
		learn_rate = total_error / 10; //dynamic learn rate is depending of the error
		if (learn_rate > 0.7) learn_rate = 0.7;

		//cout << "Max.Gradient=" << (gradients[max_gradient_y][max_gradient_x], 5) << " Change Weight=[" << max_gradient_y + "][" << max_gradient_x << "]" << " Learning Rate=" << learn_rate << " Total Error=" << total_error << endl;

		iterations_counter++;

		int result = iterations_counter % 100;//25 = every 25 iterations
		if (result == 0) {
			//runtime();
			//cout << "Trained Layer=" << used_layer << endl;
			//cout << "Iterations=" << iterations_counter << endl;
			//cout << "Total Error=" << total_error << endl;
			//cout << "Learn Rate=" << learn_rate << endl;
			//cout << "------------------------" << endl;
			cout << "*";
			used_layer++;
			if (used_layer == num_of_layers) used_layer = 0;
		}

		if (iterations_counter > maximum_iterations)break;
	} while (total_error > accepted_error);

	cout << endl;
	cout << "Needed Iterations=" << iterations_counter << endl;

	//show_weights();

	cout << "Done Learning" << endl;
	runtime();
	total_error = calc_error();
	cout << "Total Error=" << total_error << endl;
	if (total_error < accepted_error)cout << "Result OK" << endl;
	if (iterations_counter > maximum_iterations)cout << "End: Reached the maximum Iterations" << endl;
}
//-----------------------------------------------------------------------------------------------------------------
void format_the_input_data(int x, float multiplicator, float offset) { // format the input data into a range between -1 to +1

	cout << "Format Training Data:" << endl;

	for (int y = 0; y < num_of_training_data_set; y++) {
		training_data_set[y][x] *= multiplicator;
		training_data_set[y][x] += offset;
		//cout << training_data_set[y][x];
	}

	cout << "Format Test Data:" << endl;

	for (int y = 0; y < num_of_test_data_set; y++) {
		test_data_set[y][x] *= multiplicator;
		test_data_set[y][x] += offset;
		//cout << test_data_set[y][x] << endl; 
	}

	cout << "Format Data Done" << endl;
}
//-----------------------------------------------------------------------------------------------------------------
float create_random() { // random numbes beetwen -1 and +1

	float randNum = rand() % 1000;
	randNum /= 1000;
	randNum -= 0.5;
	return randNum;
}
//-----------------------------------------------------------------------------------------------------------------
void test_sigmoid_function() {

	for (float y = -10; y < 10; y += 0.1) {
		float function = sigmoid(float(y));
		cout << "Sigmoid:" << y << "/" << function << endl;
	}
}
//-----------------------------------------------------------------------------------------------------------------
void init_random_weights() { // start the learn process with random values

	cout << "Init weights by random" << endl;
	for (int y = 0; y < num_of_neurons; y++) {
		for (int x = 0; x < num_of_weights; x++) {
			weights[y][x] = create_random();
			//cout << "Weight:" << weights[y][x] << endl; 
		}
	}
}
//-----------------------------------------------------------------------------------------------------------------
void init_learned_weights() { // this is copy and paste from serial output all weights after learning process

	cout << "Init weights by learned values";
		
	weights[0][0] = -9.80669;
	weights[0][1] = -0.033;
	weights[0][2] = 11.7945;
	weights[0][3] = 0;
	weights[0][4] = -6.46065;
	weights[0][5] = 0.224;
	weights[1][0] = -1.30451;
	weights[1][1] = -0.142;
	weights[1][2] = 0.462;
	weights[1][3] = -0.036;
	weights[1][4] = 0.205;
	weights[1][5] = -0.355;
	weights[2][0] = -1.41765;
	weights[2][1] = 0.327;
	weights[2][2] = 0.461;
	weights[2][3] = -0.009;
	weights[2][4] = 1.10324;
	weights[2][5] = 0.442;
	weights[3][0] = -2.46416;
	weights[3][1] = -0.064;
	weights[3][2] = -0.809;
	weights[3][3] = -4.56174;
	weights[3][4] = -1.64992;
	weights[3][5] = 1.753;
	weights[4][0] = -0.855007;
	weights[4][1] = -0.118;
	weights[4][2] = -0.079;
	weights[4][3] = 0.216;
	weights[4][4] = 0.218;
	weights[4][5] = 0.395;
	weights[5][0] = 8.61983;
	weights[5][1] = 0.226;
	weights[5][2] = 0.271;
	weights[5][3] = 0.038;
	weights[5][4] = 0.369;
	weights[5][5] = -8.08471;
	weights[6][0] = 0.167;
	weights[6][1] = -0.794737;
	weights[6][2] = -0.465;
	weights[6][3] = 3.89399;
	weights[6][4] = 0.203;
	weights[6][5] = -1.55507;
	weights[7][0] = -0.178;
	weights[7][1] = -0.167;
	weights[7][2] = 0.173;
	weights[7][3] = 0.164;
	weights[7][4] = -0.359;
	weights[7][5] = 0.211;
	weights[8][0] = -0.247;
	weights[8][1] = 0.368;
	weights[8][2] = 0.047;
	weights[8][3] = -0.556;
	weights[8][4] = 0.162;
	weights[8][5] = 0.266564;
	weights[9][0] = -1.35495;
	weights[9][1] = 0.359;
	weights[9][2] = 0.223;
	weights[9][3] = 0.241;
	weights[9][4] = 0.029;
	weights[9][5] = 0.278;
	weights[10][0] = -0.184;
	weights[10][1] = 6.2351;
	weights[10][2] = -0.31;
	weights[10][3] = 0.342;
	weights[10][4] = -0.212;
	weights[10][5] = -4.12425;
	weights[11][0] = -0.921326;
	weights[11][1] = 0.435725;
	weights[11][2] = -0.236;
	weights[11][3] = 8.60526;
	weights[11][4] = 1.21844;
	weights[11][5] = -5.2496;
	weights[12][0] = 15.7858;
	weights[12][1] = -1.01117;
	weights[12][2] = -0.13;
	weights[12][3] = -0.15;
	weights[12][4] = -0.494;
	weights[12][5] = -11.2745;
	weights[13][0] = -0.107;
	weights[13][1] = 0.0479999;
	weights[13][2] = 0.129;
	weights[13][3] = 0.123;
	weights[13][4] = -0.416;
	weights[13][5] = -2.14834;
}
//######################################################################################################
int main()
{
	cout << "Machine Learning & Prediction" << endl;

	//             (column, multiplicator, offset)
	format_the_input_data(0, 0.2, 0); // format the input data into a range between -1 to +1
	format_the_input_data(1, 0.3, 0); // format the input data into a range between -1 to +1
	format_the_input_data(2, 0.2, 0); // format the input data into a range between -1 to +1
	format_the_input_data(3, 0.5, 0); // format the input data into a range between -1 to +1
	format_the_input_data(4, 1.0, 0); // format the input data into a range between -1 to +1

	//test_sigmoid_function();
	if (use_init_weights == 0)init_random_weights(); // start with random values > start from scratch
	if (use_init_weights == 1)init_learned_weights(); //start with allready learned values

	start_learning(); // only for learning process ( learning mode: backpropagation )
	start_predict();

	for (int r = 0; r < learn_extra_rounds; r++) { //extra rounds for improvement
		if (iterations_counter > maximum_iterations) {
			iterations_counter = 0;
			init_random_weights();
			start_learning();
			start_predict();
		}
	}

	for (;;) {
		// endless loop
	}

	return 0;
}
//######################################################################################################
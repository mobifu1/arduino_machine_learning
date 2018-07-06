//Demo KNN Classification of Fruits
//Machine Learning with Backpropagation Learning Mode

#define num_of_inputs 4   //neurons
#define num_of_hiddens 5  //neurons
#define num_of_outputs 4  //neurons
#define num_of_weights 6  //1 x neuron = input, input, input, input, input, output
#define num_of_neurons 14 //neurons
#define num_of_layers 3   //output, hidden 2, hidden 1

float inputs[num_of_inputs] = { };   //input neuron data
float outputs[num_of_outputs] = { }; //output neuron data
//--------------------------------------------------------------------
float weights[num_of_neurons][num_of_weights] = {

  //neuron: input, input, input, input, input, output

};
//--------------------------------------------------------------------
int max_gradient_x ; // x position in gradients table
int max_gradient_y ; // y position in gradients table

float gradients[num_of_neurons][num_of_weights] = {
};
//--------------------------------------------------------------------
String objects[4] = {"Tomato", "Banana", "Melone", "Raspberry"};
//--------------------------------------------------------------------
const int num_of_data_set = 8; //length of table
float training_data_set[num_of_data_set][num_of_inputs + num_of_outputs] = { //input and output data to learn
  //4x input     |4x output
  {1, 2, 0.05, 1, 1, 0, 0, 0}, // orange tomato               Input 0:   shape: 1=round, 2=oval    3=long
  {1, 1, 0.05, 1, 1, 0, 0, 0}, // red tomato                  Input 1:   color: 1=red    2=orange  3=yellow  4=lightgreen 5=green
  {3, 4, 0.20, 1, 0, 1, 0, 0}, // lightgreen banana           Input 2:  weight: in Kilogramms
  {3, 3, 0.20, 1, 0, 1, 0, 0}, // yellow banana               Input 3: surface: 1=even   2=spoty   3=rough
  {1, 3, 3.00, 2, 0, 0, 1, 0}, // yellow melone              Output 0:    type: 1=tomato
  {1, 5, 3.00, 2, 0, 0, 1, 0}, // green melone               Output 1:    type: 1=banana
  {1, 1, 0.01, 3, 0, 0, 0, 1}, // red raspberry              Output 2:    type: 1=melone
  {1, 5, 0.01, 3, 0, 0, 0, 1}, // green raspberry            Output 3:    type: 1=raspberry
};
//--------------------------------------------------------------------
const int num_of_test_data_set = 12; //length of table
float test_data_set[num_of_test_data_set][num_of_inputs] = { //iput data to predict
  {1, 5, 0.015, 1}, // Input: green tomato
  {1, 2, 0.030, 1}, // Input: orange tomato
  {1, 1, 0.050, 1}, // Input: red tomato
  {3, 5, 0.150, 1}, // Input: green banana
  {3, 4, 0.170, 1}, // Input: lightgreen banana
  {3, 3, 0.200, 1}, // Input: yellow banana
  {1, 3, 2.000, 2}, // Input: yellow melone
  {1, 5, 2.500, 2}, // Input: green melone
  {1, 4, 3.000, 2}, // Input: lightgreen melone
  {1, 1, 0.005, 3}, // Input: red raspberry
  {1, 5, 0.010, 3}, // Input: green raspberry
  {1, 3, 0.020, 3}, // Input: yellow raspberry
};
//--------------------------------------------------------------------
float learn_rate = 0; // dynamic calculation
int iterations_counter = 0;
int maximum_iterations = 1500;
float accepted_error = 0.05;
//-----------------------------------------------------------------------------------------------------------------
void setup() {

  Serial.begin(9600);
}
//-----------------------------------------------------------------------------------------------------------------
void loop() {

  Serial.println("Arduino Machine Learn & Predict");

  //test_sigmoid_function();
  init_random_weights(); // start with random values > start from scratch
  //init_learned_weights(); //start with allready learned values

  start_learning(); // only for learning process ( learning mode: backpropagation )
  start_predict();
  for (;;) {
    // endless loop
  }
}
//-----------------------------------------------------------------------------------------------------------------
void start_predict() {

  Serial.println("------------------------");
  Serial.println("Start Predict:");

  for ( int i = 0; i < num_of_test_data_set; i++) {
    calc_neuron_net(test_data_set[i][0], test_data_set[i][1], test_data_set[i][2], test_data_set[i][3]); //input test data
    Serial.print(String(i) + ". Out 0 = " + String(outputs[0]) + " / Out 1 = " + String(outputs[1]) + " / Out 2 = " + String(outputs[2]) + " / Out 3 = " + String(outputs[3]));
    if (outputs[0] >= 0.85)Serial.print(" = " + objects[0]);
    if (outputs[1] >= 0.85) Serial.print(" = " + objects[1]);
    if (outputs[2] >= 0.85) Serial.print(" = " + objects[2]);
    if (outputs[3] >= 0.85) Serial.print(" = " + objects[3]);
    Serial.println();
  }

  Serial.println("Stop  Predict");
  Serial.println("------------------------");
}
//-----------------------------------------------------------------------------------------------------------------
void start_learning() {

  Serial.println("Start Learning");
  float total_error;
  int used_layer = 0;

  do { //search for local minimum

    //Backpropagsatio: training the layers
    if (used_layer == 0)  calc_max_gradient(10, 13);     // number of output layer  = from 10 to 13 neuron
    if (used_layer == 1)  calc_max_gradient(5, 9);       // number of hidden layer 2= from 0 to 4 neuron
    if (used_layer == 2)  calc_max_gradient(0, 4);       // number of hidden layer 1= from 0 to 4 neuron

    if (gradients[max_gradient_y][max_gradient_x] > 0) weights[max_gradient_y][max_gradient_x] += learn_rate;
    if (gradients[max_gradient_y][max_gradient_x] < 0) weights[max_gradient_y][max_gradient_x] -= learn_rate;
    total_error = calc_error();

    learn_rate = total_error / 10 ; //dynamic learn_rate

    //Serial.println("Max.Gradient=" + String(gradients[max_gradient_y][max_gradient_x], 5) + " Change Weight=[" + String(max_gradient_y) + "][" + String(max_gradient_x) + "]" + " Learning Rate=" + String(learn_rate, 5) + " Total Error=" + String(total_error, 5));

    iterations_counter++;

    int result = iterations_counter % 25;//every 50 iterations
    if (result == 0) {
      used_layer++;
      if (used_layer == num_of_layers) used_layer = 0;
      Serial.println("------------------------");
      Serial.println("Iterations=" + String(iterations_counter));
      Serial.println("Total Error=" + String(total_error, 5));
      Serial.println("Learn Rate=" + String(learn_rate));
    }

    if (iterations_counter > maximum_iterations)break;
  } while (total_error > accepted_error);

  Serial.println("Needed Iterations=" + String(iterations_counter));

  show_weights();

  Serial.println("------------------------");
  Serial.println("Done Learning");
  total_error = calc_error();
  Serial.println("Total Error=" + String(total_error, 5));
  if (total_error < accepted_error)Serial.println("Result OK");
  if (iterations_counter > maximum_iterations)Serial.println("Result Fail");
}
//-----------------------------------------------------------------------------------------------------------------
void calc_max_gradient(int from_neuron, int to_neuron) { //number of hidden layer 1=from 0 to 4 neuron   /    number of hidden layer 2= from 5 to 9 neuron

  float total_error = calc_error();
  float max_steigung = 0;
  to_neuron++;

  for ( int y = from_neuron; y < to_neuron; y++) {
    for ( int x = 0; x < num_of_weights; x++) {

      weights[y][x] += learn_rate;

      float new_error = calc_error();
      float diff = total_error - new_error;

      gradients[y][x] = diff;

      //Serial.println("Gradient=" + String(diff, 5) + " Position=[" + String(y) + "][" + String(x) + "]" + " Total Error=" + String(total_error, 5));

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
float calc_error() {

  float total_error = 0;
  float error_0;
  float error_1;
  float error_2;
  float error_3;

  for (int i = 0; i < num_of_data_set; i++) {
    calc_neuron_net(training_data_set[i][0] , training_data_set[i][1] , training_data_set[i][2] , training_data_set[i][3]);

    error_0 = pow ((training_data_set[i][4] - outputs[0]), 2); //error^2
    error_1 = pow ((training_data_set[i][5] - outputs[1]), 2); //error^2
    error_2 = pow ((training_data_set[i][6] - outputs[2]), 2); //error^2
    error_3 = pow ((training_data_set[i][7] - outputs[3]), 2); //error^2

    total_error += error_0;
    total_error += error_1;
    total_error += error_2;
    total_error += error_3;
    //Serial.println("Error=" + String(total_error));
  }
  //Serial.println("Total Error=" + String(total_error));
  total_error *= 0.5;
  return total_error;
}
//-----------------------------------------------------------------------------------------------------------------
void calc_neuron_net(float x0, float x1, float x2, float x3) { //input data

  //######first hidden layer##########################
  float y0 = calc_neuron(x0, weights[0][0], x1, weights[0][1], x2, weights[0][2], x3, weights[0][3], 0, 0, weights[0][5]);
  y0 = sigmoid (y0);
  //Serial.println("Y0=" + String(y0));
  //-------------------------------------------------
  float y1 = calc_neuron(x0, weights[1][0], x1, weights[1][1], x2, weights[1][2], x3, weights[1][3], 0, 0, weights[1][5]);
  y1 = sigmoid (y1);
  //Serial.println("Y1=" + String(y1));
  //-------------------------------------------------
  float y2 = calc_neuron(x0, weights[2][0], x1, weights[2][1], x2, weights[2][2], x3, weights[2][3], 0, 0, weights[2][5]);
  y2 = sigmoid (y2);
  //Serial.println("Y2=" + String(y2));
  //-------------------------------------------------
  float y3 = calc_neuron(x0, weights[3][0], x1, weights[3][1], x2, weights[3][2], x3, weights[3][3], 0, 0, weights[3][5]);
  y3 = sigmoid (y3);
  //Serial.println("Y3=" + String(y3));
  //-------------------------------------------------
  float y4 = calc_neuron(x0, weights[4][0], x1, weights[4][1], x2, weights[4][2], x3, weights[4][3], 0, 0, weights[4][5]);
  y4 = sigmoid (y4);
  //Serial.println("Y4=" + String(y4));

  //######second hidden layer##########################
  float y00 = calc_neuron(y0, weights[5][0], y1, weights[5][1], y2, weights[5][2], y3, weights[5][3], y4, weights[5][4], weights[5][5]);
  y00 = sigmoid (y00);
  //Serial.println("y00=" + String(y00));
  //-------------------------------------------------
  float y01 = calc_neuron(y0, weights[6][0], y1, weights[6][1], y2, weights[6][2], y3, weights[6][3], y4, weights[6][4], weights[6][5]);
  y01 = sigmoid (y01);
  //Serial.println("y01=" + String(y01));
  //-------------------------------------------------
  float y02 = calc_neuron(y0, weights[7][0], y1, weights[7][1], y2, weights[7][2], y3, weights[7][3], y4, weights[7][4], weights[7][5]);
  y02 = sigmoid (y02);
  //Serial.println("y02=" + String(y02));
  //-------------------------------------------------
  float y03 = calc_neuron(y0, weights[8][0], y1, weights[8][1], y2, weights[8][2], y3, weights[8][3], y4, weights[8][4], weights[8][5]);
  y03 = sigmoid (y03);
  //Serial.println("y03=" + String(y03));
  //-------------------------------------------------
  float y04 = calc_neuron(y0, weights[9][0], y1, weights[9][1], y2, weights[9][2], y3, weights[9][3], y4, weights[9][4], weights[9][5]);
  y04 = sigmoid (y04);
  //Serial.println("y04=" + String(y04));

  //######output layer#################################
  float out0 = calc_neuron(y00, weights[10][0], y01, weights[10][1], y02, weights[10][2], y03, weights[10][3], y04, weights[10][4], weights[10][5]);
  out0 = sigmoid (out0);
  outputs[0] = out0;
  //Serial.println("Out0=" + String(out0));
  //-------------------------------------------------
  float out1 = calc_neuron(y00, weights[11][0], y01, weights[11][1], y02, weights[11][2], y03, weights[11][3], y04, weights[11][4], weights[11][5]);
  out1 = sigmoid (out1);
  outputs[1] = out1;
  //Serial.println("Out1=" + String(out1));
  //-------------------------------------------------
  float out2 = calc_neuron(y00, weights[12][0], y01, weights[12][1], y02, weights[12][2], y03, weights[12][3], y04, weights[12][4], weights[12][5]);
  out2 = sigmoid (out2);
  outputs[2] = out2;
  //Serial.println("Out2=" + String(out2));
  //-------------------------------------------------
  float out3 = calc_neuron(y00, weights[13][0], y01, weights[13][1], y02, weights[13][2], y03, weights[13][3], y04, weights[13][4], weights[13][5]);
  out3 = sigmoid (out3);
  outputs[3] = out3;
  //Serial.println("Out3=" + String(out3));
}
//-----------------------------------------------------------------------------------------------------------------
float calc_neuron(float input_0, float weight_0, float input_1, float weight_1, float input_2, float weight_2, float input_3, float weight_3, float input_4, float weight_4, float weight_output) {

  float output = weight_output + (input_0 * weight_0) + (input_1 * weight_1) + (input_2 * weight_2) + (input_3 * weight_3) + (input_4 * weight_4);
  return output;
}
//-----------------------------------------------------------------------------------------------------------------
float sigmoid(float x) { //sigmoid funktion

  float e = 2.71828; float steilheit = 2;
  float result;
  result = 1 / (1 + (pow (e, (steilheit * x * -1))));
  return result;
}
//-----------------------------------------------------------------------------------------------------------------
void test_sigmoid_function() {

  for ( float y = -10; y < 10; y += 0.1) {
    float function = sigmoid(float(y));
    Serial.println("Sigmoid:" + String(y) + "/" + String(function));
  }
}
//-----------------------------------------------------------------------------------------------------------------
void show_weights() {

  Serial.println("------------------------");
  Serial.println("Show Weights:");
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      Serial.println("weights[" + String(y) + "][" + String(x) + "] = " + String(weights[y][x], 7) + ";");
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
void init_random_weights() {

  Serial.println("Init weights by random");
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      weights[y][x] = create_random() ;
      //Serial.println("Weight:" + String(weights[y][x]));
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
void init_learned_weights() {

  Serial.println("Init weights by learned values");

  weights[0][0] = 0.3100000;
  weights[0][1] = -0.4100000;
  weights[0][2] = -0.0200000;
  weights[0][3] = -0.3900000;
  weights[0][4] = -0.2900000;
  weights[0][5] = -0.2600000;
  weights[1][0] = 1.9146098;
  weights[1][1] = -0.0900000;
  weights[1][2] = -0.4000000;
  weights[1][3] = -1.3004496;
  weights[1][4] = -0.2000000;
  weights[1][5] = -2.1343200;
  weights[2][0] = 1.8776665;
  weights[2][1] = 0.0399900;
  weights[2][2] = 0.9593400;
  weights[2][3] = -1.4246271;
  weights[2][4] = 0.1100000;
  weights[2][5] = 0.1638652;
  weights[3][0] = -0.2000000;
  weights[3][1] = -0.2900000;
  weights[3][2] = 1.8571056;
  weights[3][3] = -0.4700000;
  weights[3][4] = 0.1800000;
  weights[3][5] = -0.8590795;
  weights[4][0] = 0.3900000;
  weights[4][1] = -0.0243500;
  weights[4][2] = -1.0992579;
  weights[4][3] = 0.6443516;
  weights[4][4] = 0.1000000;
  weights[4][5] = -0.0700000;
  weights[5][0] = 0.2499999;
  weights[5][1] = -0.5200000;
  weights[5][2] = 0.4100000;
  weights[5][3] = 2.0947673;
  weights[5][4] = -0.7400000;
  weights[5][5] = -1.0400000;
  weights[6][0] = 0.2000000;
  weights[6][1] = 2.9162006;
  weights[6][2] = 0.0100000;
  weights[6][3] = -0.1300000;
  weights[6][4] = 0.3100000;
  weights[6][5] = -1.5900002;
  weights[7][0] = 0.4100000;
  weights[7][1] = -0.0900000;
  weights[7][2] = -0.3100000;
  weights[7][3] = 0.2600000;
  weights[7][4] = 0.1400000;
  weights[7][5] = -0.0500000;
  weights[8][0] = 0.0000000;
  weights[8][1] = 0.0600000;
  weights[8][2] = -4.0821800;
  weights[8][3] = 0.1100000;
  weights[8][4] = 0.9154006;
  weights[8][5] = 0.5500001;
  weights[9][0] = 0.3800000;
  weights[9][1] = -2.5313003;
  weights[9][2] = -1.1385400;
  weights[9][3] = -2.0793800;
  weights[9][4] = 0.6590644;
  weights[9][5] = 1.7255099;
  weights[10][0] = -0.7297590;
  weights[10][1] = -0.8068005;
  weights[10][2] = -0.2300000;
  weights[10][3] = -3.1058817;
  weights[10][4] = 2.4448047;
  weights[10][5] = -0.7493392;
  weights[11][0] = -0.1359398;
  weights[11][1] = 2.9837804;
  weights[11][2] = -0.2900000;
  weights[11][3] = -0.3100000;
  weights[11][4] = -0.3537177;
  weights[11][5] = -1.2900000;
  weights[12][0] = 2.1682329;
  weights[12][1] = -1.3599999;
  weights[12][2] = -0.2900000;
  weights[12][3] = -0.3500000;
  weights[12][4] = -1.1086065;
  weights[12][5] = -0.3700000;
  weights[13][0] = -0.5200000;
  weights[13][1] = -1.1099999;
  weights[13][2] = -0.1500000;
  weights[13][3] = 2.9565649;
  weights[13][4] = 0.1800000;
  weights[13][5] = -1.3313150;
}
//-----------------------------------------------------------------------------------------------------------------
float create_random() {

  float randNum = rand() % 1000;
  randNum /= 1000;
  randNum -= 0.5;
  return randNum;
}
//-----------------------------------------------------------------------------------------------------------------


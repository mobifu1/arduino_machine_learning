//Simple AI with learn and predict process, running on Arduino
//using one simple neuron with 2 inputs and 1 output line
//Learn process: 3 different combination of input values defined by input table to learn the 3 output values.
//Predict process: challenge is to predict the last output value by unlearned last input value.

/*

  x1---O
    \ / \
     X   O---Y                   O=OneNeuron
    / \ /
  x2---O

*/

const float input[4][2] = {//Objects (Attributes) X1, X2
  {0, 0},//to learn
  {0, 1},//to learn
  {1, 0},//to learn
  {1, 1},//input for predict!!!
};

const float output[4][1] = { //Output = Y  > XOR
  {0},//to learn
  {1},//to learn
  {1},//to learn
  {},//value to predict!!!
};

float weight[9] = { };//weight (w_out, w_in1, w_in2)
float gradient[9] = { };//w0-w8

float learnfactor = 0.1;
float iteration_counter = 0;
float maximum_iterations = 100000;
float accepted_error = 0.00000001;

//--------------------------------------------------------------------------------------------------------
void setup() {

  Serial.begin(9600);
}

void loop() {

  Serial.println("Arduino Machine Learn & Predict");
  init_weight();
  start_learning();
  start_predict();
  for (;;) {
    // endless loop
  }
}
//--------------------------------------------------------------------------------------------------------
void start_predict() {

  Serial.println("------------------------");
  Serial.println("Start Predict:");

  for (int i = 0; i < 4; i++) {
    float x1 = input[i][0];
    float x2 = input[i][1];
    float out = calc_neuron_net(x1, x2);
    //Serial.println("Out=" + String(out));
    Serial.println("X1=" + String(x1) + " X2=" + String(x2) + " >  Y=" + String(output[i][0]));
  }

  Serial.println("Stop  Predict");
  Serial.println("------------------------");
}
//--------------------------------------------------------------------------------------------------------
void start_learning() {

  Serial.println("Start Learning");
  float total_error;

  do { //search for local minimum
    int position = calc_gradient();
    //Serial.println("Max. Steigung Gewicht=[" + String(position) + "]");
    if (gradient[position] > 0) weight[position] += learnfactor;
    if (gradient[position] < 0) weight[position] -= learnfactor;
    total_error = calc_error();
    //Serial.println("Total Error=" + String(total_error));
    iteration_counter++;
    if (iteration_counter > maximum_iterations)break;
  } while (total_error > accepted_error);

  Serial.println("Needed Iterations=" + String(iteration_counter));
  Serial.println("------------------------");
  Serial.println("Calculated weight:");

  for (int i = 0; i < 9; i++) {
    Serial.println("w" + String(i) + "=" + String(weight[i]));
  }

  Serial.println("------------------------");
  Serial.println("Done Learning");
  total_error = calc_error();
  Serial.println("Total Error=" + String(total_error));
  if (total_error < accepted_error)Serial.println("Result OK");
  if (iteration_counter > maximum_iterations)Serial.println("Result Fail");
}//-------------------------------------------------------------------------------------------------------
int calc_gradient() {

  float total_error = calc_error();
  //Serial.println("Total Error=" + String(total_error));
  float max_steigung = 0;
  int position;

  for (int index = 0 ; index < 9; index++) {
    weight[index] += learnfactor;
    float new_error = calc_error();
    float diff = total_error - new_error;
    gradient[index] = diff;
    //Serial.println("Error Anteil w[" + String(index) + "]=" + String(diff));
    weight[index] -= learnfactor;

    if (fabs(diff) > max_steigung) {
      max_steigung = fabs(diff);
      position = index;
    }
  }
  return position;
}
//--------------------------------------------------------------------------------------------------------
float calc_error() {

  float total_error = 0;
  float error;
  for (int i = 0; i < 3; i++) {
    float x1 = input[i][0]; float x2 = input[i][1];
    //Serial.println("------------------------");
    //Serial.println("X1=" + String(x1) + " X2=" + String(x2));
    float out = calc_neuron_net(x1, x2);
    //Serial.println("Out=" + out + " [" + output[i][0] + "]");
    error = pow ((output[i][0] - out), 2);//error^2
    total_error += error;
    //Serial.println("Error=" + String(error));
  }
  //Serial.println("Total Error=" + String(total_error));
  total_error *= 0.5;
  return total_error;
}
//--------------------------------------------------------------------------------------------------------
float calc_neuron_net(float x1, float x2) {

  float y1 = calc_neuron(x1, weight[1], x2, weight[2], weight[0]);
  y1 = sigmoid (y1);
  //Serial.println("Y1=" + String(y1));
  //-------------------------------------------------
  float y2 = calc_neuron(x1, weight[4], x2, weight[5], weight[3]);
  y2 = sigmoid (y2);
  //Serial.println("Y2=" + String(y2));
  //-------------------------------------------------
  float out = calc_neuron(y1, weight[7], y2, weight[8], weight[6]);
  out = sigmoid (out);
  return out;
}
//--------------------------------------------------------------------------------------------------------
float calc_neuron(float input_1, float weight_1, float input_2, float weight_2, float weight_output) {

  float output = weight_output + (input_1 * weight_1) + (input_2 * weight_2);
  return output;
}
//--------------------------------------------------------------------------------------------------------
float sigmoid(float x) { //sigmoid funktion

  float e = 2.71828; float steilheit = 2;
  float result;
  result = 1 / (1 + (pow (e, (steilheit * x * -1))));
  return result;
}
//--------------------------------------------------------------------------------------------------------
void init_weight() {

  Serial.println("Init weight by random");
  for ( int z = 0; z < 10; z++) {
    weight[z] = create_random();
    //Serial.println("W" + String(z) + ":" + String(weight[z]));
  }
}
//--------------------------------------------------------------------------------------------------------
float create_random() {

  float randNum = rand() % 1000;
  randNum /= 1000;
  return randNum;
}
//--------------------------------------------------------------------------------------------------------


//Demo KNN Classification of Fruits

#define num_of_inputs 4
#define num_of_hiddens 3
#define num_of_outputs 2
#define num_of_weights 5
#define num_of_neurons 5

float inputs[num_of_inputs] = { };   //input neuron data
float outputs[num_of_outputs] = { }; //output neuron data
//--------------------------------------------------------------------
float weights[num_of_neurons][num_of_weights] = {

  //neuron: input, input, input, input, output

};
//--------------------------------------------------------------------
int max_gradient_x ; // x position in gradients table
int max_gradient_y ; // y position in gradients table

float gradients[num_of_neurons][num_of_weights] = {
};
//--------------------------------------------------------------------
String objects[2] = {"Tomato", "Banana"};
//--------------------------------------------------------------------
const int num_of_data_set = 4; //length of table
float training_data_set[num_of_data_set][num_of_inputs + num_of_outputs] = { //input and output data to learn
  //4x input |2x output
  {1, 2, 2, 1, 1, 0}, //orange tomato               Input 0:   shape: 1=round, 2=oval    3=long
  {1, 1, 2, 1, 1, 0}, //red tomato                  Input 1:   color: 1=red    2=orange  3=yellow  4=lightgreen 5=green
  {3, 4, 3, 1, 0, 1}, //lightgreen banana           Input 2:  weight: 1=small  2=light   3=middle  4=haevy
  {3, 3, 3, 1, 0, 1}, //yellow banana               Input 3: surface: 1=even   2=spoty   3=rough
  //                                                Output 0:    type: 1=tomato
  //                                                Output 1:    type: 1=banana
};
//--------------------------------------------------------------------
const int num_of_test_data_set = 6; //length of table
float test_data_set[num_of_test_data_set][num_of_inputs] = { //iput data to predict
  {1, 3, 2, 1}, // Input: yellow tomato
  {1, 2, 2, 1}, // Input: orange tomato
  {1, 1, 2, 1}, // Input: red tomato
  {3, 5, 3, 1}, // Input: green banana
  {3, 4, 3, 1}, // Input: lightgreen banana
  {3, 3, 3, 1}, // Input: yellow banana
};
//--------------------------------------------------------------------
float learn_rate ; //calculated by error
int iterations_counter = 0;
int maximum_iterations = 500;
float accepted_error = 0.0001;
//-----------------------------------------------------------------------------------------------------------------
void setup() {

  Serial.begin(9600);
}
//-----------------------------------------------------------------------------------------------------------------
void loop() {

  Serial.println("Arduino Machine Learn & Predict");
  init_weights();
  start_learning();
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
    Serial.print("Out 0 = " + String(outputs[0]) + " / Out 1 = " + String(outputs[1]) + " = ");
    if (outputs[0] >= outputs[1]) Serial.println(objects[0]);
    if (outputs[1] >= outputs[0]) Serial.println(objects[1]);
  }

  Serial.println("Stop  Predict");
  Serial.println("------------------------");
}
//-----------------------------------------------------------------------------------------------------------------
void start_learning() {

  Serial.println("Start Learning");
  float total_error;

  do { //search for local minimum

    calc_max_gradient();
    if (gradients[max_gradient_y][max_gradient_x] > 0) weights[max_gradient_y][max_gradient_x] += learn_rate;
    if (gradients[max_gradient_y][max_gradient_x] < 0) weights[max_gradient_y][max_gradient_x] -= learn_rate;
    total_error = calc_error();
    //Serial.println("Total Error=" + String(total_error));

    learn_rate = total_error / 10; //dynamic learn_rate
    if (learn_rate < 0.1 ) learn_rate = 0.1;

    iterations_counter++;

    int result = iterations_counter % 50;
    if (result == 0) {
      Serial.println("------------------------");
      Serial.println("Iterations=" + String(iterations_counter));
      Serial.println("Total Error=" + String(total_error, 5));
      Serial.println("Learn Rate=" + String(learn_rate));
    }

    if (iterations_counter > maximum_iterations)break;
  } while (total_error > accepted_error);

  Serial.println("Needed Iterations=" + String(iterations_counter));

  //show_weights();

  Serial.println("------------------------");
  Serial.println("Done Learning");
  total_error = calc_error();
  Serial.println("Total Error=" + String(total_error, 5));
  if (total_error < accepted_error)Serial.println("Result OK");
  if (iterations_counter > maximum_iterations)Serial.println("Result Fail");
}
//-----------------------------------------------------------------------------------------------------------------
void calc_max_gradient() {

  float total_error = calc_error();
  float max_steigung = 0;

  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {

      weights[y][x] += learn_rate;

      float new_error = calc_error();
      float diff = total_error - new_error;
      gradients[y][x] = diff;

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
  for (int i = 0; i < num_of_data_set; i++) {
    calc_neuron_net(training_data_set[i][0] , training_data_set[i][1] , training_data_set[i][2] , training_data_set[i][3] );
    error_0 = pow ((training_data_set[i][4] - outputs[0]), 2); //error^2
    error_1 = pow ((training_data_set[i][5] - outputs[1]), 2); //error^2
    total_error += error_0;
    total_error += error_1;
    //Serial.println("Error=" + String(error));
  }
  //Serial.println("Total Error=" + String(total_error));
  total_error *= 0.5;
  return total_error;
}
//-----------------------------------------------------------------------------------------------------------------
void calc_neuron_net(float x0, float x1, float x2, float x3) { //input data

  float y0 = calc_neuron(x0, weights[0][0], x1, weights[0][1], x2, weights[0][2], x3, weights[0][3], weights[0][4]);
  y0 = sigmoid (y0);
  //Serial.println("Y0=" + String(y0));
  //-------------------------------------------------
  float y1 = calc_neuron(x0, weights[1][0], x1, weights[1][1], x2, weights[1][2], x3, weights[1][3], weights[1][4]);
  y1 = sigmoid (y1);
  //Serial.println("Y1=" + String(y1));
  //-------------------------------------------------
  float y2 = calc_neuron(x0, weights[2][0], x1, weights[2][1], x2, weights[2][2], x3, weights[2][3], weights[2][4]);
  y2 = sigmoid (y2);
  //Serial.println("Y2=" + String(y2));
  //-------------------------------------------------
  float out0 = calc_neuron(y0, weights[3][0], y1, weights[3][1], y2, weights[3][2], 0, 0, weights[3][4]);
  out0 = sigmoid (out0);
  outputs[0] = out0;
  //Serial.println("Out0=" + String(out0));
  //-------------------------------------------------
  float out1 = calc_neuron(y0, weights[4][0], y1, weights[4][1], y2, weights[4][2], 0, 0, weights[4][4]);
  out1 = sigmoid (out1);
  outputs[1] = out1;
  //Serial.println("Out1=" + String(out1));
}
//-----------------------------------------------------------------------------------------------------------------
float calc_neuron(float input_1, float weight_1, float input_2, float weight_2, float input_3, float weight_3, float input_4, float weight_4, float weight_output) {

  float output = weight_output + (input_1 * weight_1) + (input_2 * weight_2) + (input_3 * weight_3) + (input_4 * weight_4);
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
void show_weights() {

  Serial.println("------------------------");
  Serial.println("Show Weights:");
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      Serial.println("Weight:" + String(weights[y][x]));
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
void init_weights() {

  Serial.println("Init weights by random");
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      weights[y][x] = create_random();
      //Serial.println("Weight:" + String(weights[y][x]));
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
float create_random() {

  float randNum = rand() % 1000;
  randNum /= 1000;
  return randNum;
}
//-----------------------------------------------------------------------------------------------------------------

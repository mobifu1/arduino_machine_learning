//###########################################################################
//####       Demo ANN Classification of Fruits & Vegetables              ####
//####     Machine Learning with Backpropagation Learning Mode           ####
//####        Make what you want with this sourcecode :-)                ####
//####            Thinking about: shit in > shit out                     ####
//###########################################################################

#define num_of_inputs 5   // neurons
#define num_of_hiddens 5  // neurons
#define num_of_outputs 4  // neurons
#define num_of_weights 6  // 1 x neuron = input, input, input, input, input, bias
#define num_of_neurons 14 // total neurons
#define num_of_layers 3   // output, hidden 2, hidden 1

float inputs[num_of_inputs] = { };   // input neuron data
float outputs[num_of_outputs] = { }; // output neuron data
String objects[num_of_outputs] = {"Tomato", "Banana", "Melon", "Raspberry"}; // alias of the 4 output neurons
//--------------------------------------------------------------------
float weights[num_of_neurons][num_of_weights] = { //contains all weights and bias values

  // 1 x neuron = input, input, input, input, input, output

};
//--------------------------------------------------------------------
int max_gradient_x ; // x position in gradients table
int max_gradient_y ; // y position in gradients table

float gradients[num_of_neurons][num_of_weights] = { // values to search the highest improvement of error
};
//--------------------------------------------------------------------
const int num_of_training_data_set = 8; //length of table
float training_data_set[num_of_training_data_set][num_of_inputs + num_of_outputs] = { //input and output data to learn
  //5x input           |4x output
  {1, 2, 0.030, 1, 2.0, 1, 0, 0, 0}, // orange tomato              | Input 0 | Input 1      | Input 2    | Input 3 | Input 4     | Output 0  | Output 1 | Output 2  | Output 3    |
  {1, 1, 0.050, 1, 2.5, 1, 0, 0, 0}, // red tomato                 | 1=round | 1=red        | weight in  | 1=even  | Length in   | 1=tomato  | 1=banana | 1=melon   | 1=raspberry |
  {3, 4, 0.200, 1,  18, 0, 1, 0, 0}, // lightgreen banana          | 2=oval  | 2=orange     | kilogramms | 2=spoty | centimeters |           |          |           |             |
  {3, 3, 0.200, 1,  20, 0, 1, 0, 0}, // yellow banana              | 3=long  | 3=yellow     |            | 3=rough |             |           |          |           |             |
  {1, 5, 2.500, 2,  16, 0, 0, 1, 0}, // green melon                | 4=bulp  | 4=lightgreen |            | 4=hairy |             |           |          |           |             |
  {1, 3, 3.000, 2,  18, 0, 0, 1, 0}, // yellow melon               |         | 5=green      |            |         |             |           |          |           |             |
  {2, 5, 0.002, 3, 0.5, 0, 0, 0, 1}, // green raspberry            |         | 6=brown      |            |         |             |           |          |           |             |
  {2, 3, 0.005, 3, 1.0, 0, 0, 0, 1}, // red raspberry              |         |              |            |         |             |           |          |           |             |
};
//--------------------------------------------------------------------
const int num_of_test_data_set = 7; // length of table, count of object
float test_data_set[num_of_test_data_set][num_of_inputs] = { // input data to predict
  //5x input
  {1, 3, 0.025, 1,  1.8}, // Input: round yellow tomato
  {2, 4, 0.020, 1,  1.5}, // Input: small oval lightgreen tomato
  {3, 5, 0.150, 1,   12}, // Input: small green banana
  {1, 4, 2.000, 2,   14}, // Input: round lightgreen melon
  {2, 5, 2.300, 2,   16}, // Input: oval green melon
  {2, 1, 0.005, 3,  1.5}, // Input: oval big red raspberry
  {2, 2, 0.003, 3, 0.75}, // Input: oval orange raspberry
};
//######################################################################################################
int activation_function = 0;  // 0=sigmoid > OUTPUT:(0 to 1), 1=tanh OUTPUT:(-1 to 1), 2=relu OUTPUT:(0 to max)
float learn_rate = 0; // dynamic calculation
int iterations_counter = 0;
int maximum_iterations = 2000;
float accepted_error = 0.05;
int learn_extra_rounds = 3; //if the software cant find a global minimum > init the weights new.
float threshold = 0.85; //for a good prediction
unsigned long time;
byte use_init_weights = 0;// 0 = init by random weights, 1 = init by learned weights
float input_data_min = 0;//format the input data in the range between min an max
float input_data_max = 1;
//######################################################################################################
void setup() {

  Serial.begin(9600);
  Serial.println(F("Arduino Machine Learn & Predict"));

  format_the_input_data(input_data_min, input_data_max); // format the input data into a range between 0 to +1

  //test_sigmoid_function();
  if (use_init_weights == 0)init_random_weights(); // start with random values > start from scratch
  if (use_init_weights == 1)init_learned_weights(); //start with allready learned values
}
//-----------------------------------------------------------------------------------------------------------------
void loop() {

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
}
//-----------------------------------------------------------------------------------------------------------------
void start_predict() {

  Serial.println(F("------------------------"));
  Serial.println(F("Start Predict:"));

  Serial.println(F("Training Data:"));

  for (int i = 0; i < num_of_training_data_set; i++) {

    calc_neuron_net(training_data_set[i][0], training_data_set[i][1], training_data_set[i][2], training_data_set[i][3], training_data_set[i][4]); //input training data

    Serial.print(" Out 0 = " + String(outputs[0]));
    Serial.print(" / Out 1 = " + String(outputs[1]));
    Serial.print(" / Out 2 = " + String(outputs[2]));
    Serial.print(" / Out 3 = " + String(outputs[3]));

    Serial.print(" = " + max_classification());
    Serial.println();
  }

  Serial.println(F("Test Data:"));

  for (int i = 0; i < num_of_test_data_set; i++) {

    calc_neuron_net(test_data_set[i][0], test_data_set[i][1], test_data_set[i][2], test_data_set[i][3], test_data_set[i][4]); //input test data

    Serial.print(" Out 0 = " + String(outputs[0]));
    Serial.print(" / Out 1 = " + String(outputs[1]));
    Serial.print(" / Out 2 = " + String(outputs[2]));
    Serial.print(" / Out 3 = " + String(outputs[3]));

    Serial.print(" = " + max_classification());
    Serial.println();
  }

  Serial.println(F("Stop  Predict"));
  Serial.println(F("------------------------"));
}
//-----------------------------------------------------------------------------------------------------------------
String max_classification() {

  float result = 0;
  String alias;

  for (int i = 0; i < num_of_outputs; i++) {
    if (outputs[i] > result) {
      result = outputs[i];
      alias = objects[i];
    }
  }

  if (result < threshold) alias = F("n/a");
  return alias;
}
//-----------------------------------------------------------------------------------------------------------------
void start_learning() {

  Serial.println(F("Start Learning"));
  Serial.println(F("------------------------"));
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
    learn_rate = total_error / 10 ; //dynamic learn rate is depending of the error
    if (learn_rate > 0.7) learn_rate = 0.7;

    Serial.println("Max.Gradient=" + String(gradients[max_gradient_y][max_gradient_x], 5) + " Change Weight=[" + String(max_gradient_y) + "][" + String(max_gradient_x) + "]" + " Learning Rate=" + String(learn_rate, 5) + " Total Error=" + String(total_error, 5));

    iterations_counter++;

    int result = iterations_counter % 25;//every 25 iterations
    if (result == 0) {
      runtime();
      Serial.println("Trained Layer=" + String(used_layer));
      Serial.println("Iterations=" + String(iterations_counter));
      Serial.println("Total Error=" + String(total_error, 5));
      Serial.println("Learn Rate=" + String(learn_rate, 5));
      Serial.println("------------------------");

      used_layer++;
      if (used_layer == num_of_layers) used_layer = 0;
    }

    if (iterations_counter > maximum_iterations)break;
  } while (total_error > accepted_error);

  Serial.println("Needed Iterations=" + String(iterations_counter));

  show_weights();

  Serial.println(F("Done Learning"));
  runtime();
  total_error = calc_error();
  Serial.println("Total Error=" + String(total_error, 5));
  if (total_error < accepted_error)Serial.println(F("Result OK"));
  if (iterations_counter > maximum_iterations)Serial.println(F("End: Reached the maximum Iterations"));
}
//-----------------------------------------------------------------------------------------------------------------
void calc_max_gradient(int from_neuron, int to_neuron) { // search max. gradient of one layer

  float total_error = calc_error();
  float max_steigung = 0;

  for ( int y = from_neuron; y <= to_neuron; y++) {
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
float calc_error() { // find the whole error of neuron network by input the training data

  float total_error = 0;
  float error_0;
  float error_1;
  float error_2;
  float error_3;

  for (int i = 0; i < num_of_training_data_set; i++) {

    calc_neuron_net(training_data_set[i][0] , training_data_set[i][1] , training_data_set[i][2] , training_data_set[i][3] , training_data_set[i][4]);

    error_0 = pow ((training_data_set[i][5] - outputs[0]), 2);
    error_1 = pow ((training_data_set[i][6] - outputs[1]), 2);
    error_2 = pow ((training_data_set[i][7] - outputs[2]), 2);
    error_3 = pow ((training_data_set[i][8] - outputs[3]), 2);

    total_error += error_0;
    total_error += error_1;
    total_error += error_2;
    total_error += error_3;
  }

  //Serial.println("Total Error=" + String(total_error));
  total_error *= 0.5;
  return total_error;
}
//-----------------------------------------------------------------------------------------------------------------
void calc_neuron_net(float x0, float x1, float x2, float x3, float x4) { // calculate all output values by input values

  //######first hidden layer##########################
  float y0 = calc_neuron(x0, weights[0][0], x1, weights[0][1], x2, weights[0][2], x3, weights[0][3], x4, weights[0][4], weights[0][5]);
  y0 = activation_functions (y0);
  //Serial.println("Y0=" + String(y0));
  //-------------------------------------------------
  float y1 = calc_neuron(x0, weights[1][0], x1, weights[1][1], x2, weights[1][2], x3, weights[1][3], x4, weights[1][4], weights[1][5]);
  y1 = activation_functions (y1);
  //Serial.println("Y1=" + String(y1));
  //-------------------------------------------------
  float y2 = calc_neuron(x0, weights[2][0], x1, weights[2][1], x2, weights[2][2], x3, weights[2][3], x4, weights[2][4], weights[2][5]);
  y2 = activation_functions (y2);
  //Serial.println("Y2=" + String(y2));
  //-------------------------------------------------
  float y3 = calc_neuron(x0, weights[3][0], x1, weights[3][1], x2, weights[3][2], x3, weights[3][3], x4, weights[3][4], weights[3][5]);
  y3 = activation_functions (y3);
  //Serial.println("Y3=" + String(y3));
  //-------------------------------------------------
  float y4 = calc_neuron(x0, weights[4][0], x1, weights[4][1], x2, weights[4][2], x3, weights[4][3], x4, weights[4][4], weights[4][5]);
  y4 = activation_functions (y4);
  //Serial.println("Y4=" + String(y4));

  //######second hidden layer##########################
  float y00 = calc_neuron(y0, weights[5][0], y1, weights[5][1], y2, weights[5][2], y3, weights[5][3], y4, weights[5][4], weights[5][5]);
  y00 = activation_functions (y00);
  //Serial.println("y00=" + String(y00));
  //-------------------------------------------------
  float y01 = calc_neuron(y0, weights[6][0], y1, weights[6][1], y2, weights[6][2], y3, weights[6][3], y4, weights[6][4], weights[6][5]);
  y01 = activation_functions (y01);
  //Serial.println("y01=" + String(y01));
  //-------------------------------------------------
  float y02 = calc_neuron(y0, weights[7][0], y1, weights[7][1], y2, weights[7][2], y3, weights[7][3], y4, weights[7][4], weights[7][5]);
  y02 = activation_functions (y02);
  //Serial.println("y02=" + String(y02));
  //-------------------------------------------------
  float y03 = calc_neuron(y0, weights[8][0], y1, weights[8][1], y2, weights[8][2], y3, weights[8][3], y4, weights[8][4], weights[8][5]);
  y03 = activation_functions (y03);
  //Serial.println("y03=" + String(y03));
  //-------------------------------------------------
  float y04 = calc_neuron(y0, weights[9][0], y1, weights[9][1], y2, weights[9][2], y3, weights[9][3], y4, weights[9][4], weights[9][5]);
  y04 = activation_functions (y04);
  //Serial.println("y04=" + String(y04));

  //######output layer#################################
  float out0 = calc_neuron(y00, weights[10][0], y01, weights[10][1], y02, weights[10][2], y03, weights[10][3], y04, weights[10][4], weights[10][5]);
  out0 = activation_functions (out0);
  outputs[0] = out0;
  //Serial.println("Out0=" + String(out0));
  //-------------------------------------------------
  float out1 = calc_neuron(y00, weights[11][0], y01, weights[11][1], y02, weights[11][2], y03, weights[11][3], y04, weights[11][4], weights[11][5]);
  out1 = activation_functions (out1);
  outputs[1] = out1;
  //Serial.println("Out1=" + String(out1));
  //-------------------------------------------------
  float out2 = calc_neuron(y00, weights[12][0], y01, weights[12][1], y02, weights[12][2], y03, weights[12][3], y04, weights[12][4], weights[12][5]);
  out2 = activation_functions (out2);
  outputs[2] = out2;
  //Serial.println("Out2=" + String(out2));
  //-------------------------------------------------
  float out3 = calc_neuron(y00, weights[13][0], y01, weights[13][1], y02, weights[13][2], y03, weights[13][3], y04, weights[13][4], weights[13][5]);
  out3 = activation_functions (out3);
  outputs[3] = out3;
  //Serial.println("Out3=" + String(out3));
}
//-----------------------------------------------------------------------------------------------------------------
float calc_neuron(float input_0, float weight_0, float input_1, float weight_1, float input_2, float weight_2, float input_3, float weight_3, float input_4, float weight_4, float weight_output) { // this is one neuron

  float output = weight_output + (input_0 * weight_0) + (input_1 * weight_1) + (input_2 * weight_2) + (input_3 * weight_3) + (input_4 * weight_4);
  return output;
}
//-----------------------------------------------------------------------------------------------------------------
void format_the_input_data(float d_1, float d_2) { // format the input data into a range between 0 to +1

  float value_min[num_of_inputs] = {};
  float value_max[num_of_inputs] = {};

  for ( int i = 0; i < num_of_inputs; i++) {
    value_min[i] = 999999;
    value_max[i] = -999999;
  }

  for ( int z = 0; z < num_of_inputs; z++) {
    for ( int y = 0; y < num_of_training_data_set; y++) {
      if (training_data_set[y][z] < value_min[z])value_min[z] = training_data_set[y][z];
      if (training_data_set[y][z] > value_max[z])value_max[z] = training_data_set[y][z];
    }
    //Serial.println("Value Min=" + String(value_min[z], 5));
    //Serial.println("Value Max=" + String(value_max[z], 5));
  }

  Serial.println(F("Format Training Data:"));

  for ( int z = 0; z < num_of_inputs; z++) {
    for ( int y = 0; y < num_of_training_data_set; y++) {
      if (value_max[z] != value_min[z]) {
        training_data_set[y][z] = (((training_data_set[y][z] - value_min[z]) * (d_2 - d_1)) / (value_max[z] - value_min[z])) + d_1;
      }
      //Serial.println(String(training_data_set[y][z], 5));

    }
    //Serial.println("---------");
  }

  Serial.println(F("Format Test Data:"));

  for ( int z = 0; z < num_of_inputs; z++) {
    for ( int y = 0; y < num_of_test_data_set; y++) {
      if (value_max[z] != value_min[z]) {
        test_data_set[y][z] = (((test_data_set[y][z] - value_min[z]) * (d_2 - d_1)) / (value_max[z] - value_min[z])) + d_1;
      }
      //Serial.println(String(test_data_set[y][z], 5));
    }
    //Serial.println("---------");
  }
}
//-----------------------------------------------------------------------------------------------------------------
float activation_functions(float x) {

  float result;
  if (activation_function == 0)result = sigmoid(x);
  if (activation_function == 1)result = tangens_hyperbolic(x);
  if (activation_function == 2)result = relu(x);
  return result;
}
//-----------------------------------------------------------------------------------------------------------------
float sigmoid(float x) { //sigmoid funktion > activation between 0 to 1

  float e = 2.71828; float gain = 2;
  float result;
  result = 1 / (1 + (pow (e, (gain * x * -1))));
  return result;
}
//-----------------------------------------------------------------------------------------------------------------
float tangens_hyperbolic(float x) { //tanh function > activation between -1 to 1

  float result; float gain = 0.5;
  result = tanh (gain * x ); //gain = 0.5 -1.5
  return result;
}
//-----------------------------------------------------------------------------------------------------------------
float relu(float x) { //relu function > activation between 0 to max

  if (x < 0)x = 0;
  return x;
}
//-----------------------------------------------------------------------------------------------------------------
void show_weights() { //you can copy the serial output of weights into the sourcecode

  Serial.println(F("------------------------"));
  Serial.println(F("Show Weights:"));
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      Serial.println("weights[" + String(y) + "][" + String(x) + "] = " + String(weights[y][x], 7) + ";");
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
void init_random_weights() { // start your lern process with random values

  Serial.println(F("Init weights by random"));
  for ( int y = 0; y < num_of_neurons; y++) {
    for ( int x = 0; x < num_of_weights; x++) {
      weights[y][x] = create_random() ;
      //Serial.println("Weight:" + String(weights[y][x]));
    }
  }
}
//-----------------------------------------------------------------------------------------------------------------
void init_learned_weights() { // this is copy and paste from serial output all weights after learning process

  Serial.println(F("Init weights by learned values"));

  weights[0][0] = 0.3070000;
  weights[0][1] = -0.4110000;
  weights[0][2] = -0.0190000;
  weights[0][3] = 0.6698763;
  weights[0][4] = -4.7596560;
  weights[0][5] = 1.7788529;
  weights[1][0] = 0.3000000;
  weights[1][1] = -0.0334202;
  weights[1][2] = -1.7732860;
  weights[1][3] = -4.6559601;
  weights[1][4] = 0.5719578;
  weights[1][5] = 1.5196481;
  weights[2][0] = 0.6398958;
  weights[2][1] = 0.0220000;
  weights[2][2] = 0.1830001;
  weights[2][3] = 0.3304780;
  weights[2][4] = 0.8886317;
  weights[2][5] = -0.5657277;
  weights[3][0] = -0.5545579;
  weights[3][1] = 0.2720000;
  weights[3][2] = 0.6831087;
  weights[3][3] = -0.6559621;
  weights[3][4] = 4.1430759;
  weights[3][5] = -1.5531576;
  weights[4][0] = 0.3879999;
  weights[4][1] = -0.1430000;
  weights[4][2] = -0.1370000;
  weights[4][3] = -0.7693543;
  weights[4][4] = 0.1000001;
  weights[4][5] = -0.0690000;
  weights[5][0] = 1.9320006;
  weights[5][1] = -3.6710000;
  weights[5][2] = 0.4120000;
  weights[5][3] = -1.9166203;
  weights[5][4] = -0.2420000;
  weights[5][5] = -0.3150000;
  weights[6][0] = 0.1990002;
  weights[6][1] = 5.2787738;
  weights[6][2] = 0.0150000;
  weights[6][3] = -0.4165656;
  weights[6][4] = 0.3150000;
  weights[6][5] = -2.4581294;
  weights[7][0] = 1.8088355;
  weights[7][1] = -0.0920000;
  weights[7][2] = -0.3110000;
  weights[7][3] = 0.2610000;
  weights[7][4] = 0.1359999;
  weights[7][5] = -1.0577508;
  weights[8][0] = 1.7511318;
  weights[8][1] = 0.0570000;
  weights[8][2] = -0.4430000;
  weights[8][3] = -1.8508313;
  weights[8][4] = 0.2320000;
  weights[8][5] = 0.1560001;
  weights[9][0] = 3.3128457;
  weights[9][1] = -0.4260000;
  weights[9][2] = -0.2880000;
  weights[9][3] = -2.7947559;
  weights[9][4] = -0.2500000;
  weights[9][5] = 0.4973691;
  weights[10][0] = -1.2907034;
  weights[10][1] = 1.4804363;
  weights[10][2] = -0.2310000;
  weights[10][3] = 0.1799999;
  weights[10][4] = 2.5232320;
  weights[10][5] = -2.7248242;
  weights[11][0] = 0.0210000;
  weights[11][1] = 2.5348809;
  weights[11][2] = -0.2880000;
  weights[11][3] = -0.3110000;
  weights[11][4] = -2.2629516;
  weights[11][5] = -1.2203776;
  weights[12][0] = -0.2090000;
  weights[12][1] = -2.6359692;
  weights[12][2] = -0.2910000;
  weights[12][3] = -0.3020000;
  weights[12][4] = -1.9255922;
  weights[12][5] = 1.2147701;
  weights[13][0] = 1.0635144;
  weights[13][1] = -1.9355142;
  weights[13][2] = -0.1530000;
  weights[13][3] = 0.4739997;
  weights[13][4] = 1.4908051;
  weights[13][5] = -1.4127073;
}
//-----------------------------------------------------------------------------------------------------------------
float create_random() { // random numbes beetwen -1 and +1

  float randNum = rand() % 1000;
  randNum /= 1000;
  randNum -= 0.5;
  return randNum;
}
//-----------------------------------------------------------------------------------------------------------------
void runtime() {

  unsigned long time = millis() / 1000;
  Serial.print(F("Runtime="));
  if (time < 60) {
    Serial.println(String(time) + "sec");
  }
  else {
    float minute = float(time) / 60;
    Serial.println(String(minute) + "min");
  }
}
//-----------------------------------------------------------------------------------------------------------------


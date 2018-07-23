//###########################################################################
//####                    Demo K-Means Clustering                        ####
//####                 unsupervised machine learning                     ####
//####         Make what you want with this sourcecode :-)               ####
//####            Thinking about: shit in > shit out                     ####
//###########################################################################

//https://www.youtube.com/watch?v=_S5tvagaQRU
//https://www.youtube.com/watch?v=wt-X61BnUCA
//https://www.youtube.com/watch?v=4b5d3muPQmA
//https://de.wikipedia.org/wiki/K-Means-Algorithmus#Variationen

#define input_data 48
#define attributes 4
const int  k = 3; //number of clusters to divide the input data

//Elbow-Plot X,Y:                                                                                                        |Y
//K=0 > Variation=0                                                                                  alfa                |
//K=1 > Variation=96.39.  alfa=89.405605                                                              /                  |
//K=2 > Variation=198.49  alfa=89.422703  diff= 0.017098                                             /                   |
//K=3 > Variation=245.34  alfa=89.299426  diff=−0.123277 > optimum!!!                               /                    |
//K=4 > Variation=279.51  alfa=89,180109  diff=−0.119317                                         X /_____________________|
//K=5 > Variation=328.03  alfa=89.126660  diff=−0.053449
//-------------------------------------------------------------------------------------------------------------------------------------------------
float training_data_set[input_data][attributes] = { //input data to clustering

  {5.1, 3.5, 1.4, 0.2},
  {4.9, 3.0, 1.4, 0.2},
  {4.7, 3.2, 1.3, 0.2},
  {4.6, 3.1, 1.5, 0.2},
  {5.0, 3.6, 1.4, 0.2},
  {5.4, 3.9, 1.7, 0.4},
  {4.6, 3.4, 1.4, 0.3},
  {5.0, 3.4, 1.5, 0.2},
  {4.4, 2.9, 1.4, 0.2},
  {4.9, 3.1, 1.5, 0.1},
  {5.4, 3.7, 1.5, 0.2},
  {4.8, 3.4, 1.6, 0.2},
  {4.8, 3.0, 1.4, 0.1},
  {4.3, 3.0, 1.1, 0.1},
  {5.8, 4.0, 1.2, 0.2},
  {5.7, 4.4, 1.5, 0.4},
  //
  {7.0, 3.2, 4.7, 1.4},
  {6.4, 3.2, 4.5, 1.5},
  {6.9, 3.1, 4.9, 1.5},
  {5.5, 2.3, 4.0, 1.3},
  {6.5, 2.8, 4.6, 1.5},
  {5.7, 2.8, 4.5, 1.3},
  {6.3, 3.3, 4.7, 1.6},
  {4.9, 2.4, 3.3, 1.0},
  {6.6, 2.9, 4.6, 1.3},
  {5.2, 2.7, 3.9, 1.4},
  {5.0, 2.0, 3.5, 1.0},
  {5.9, 3.0, 4.2, 1.5},
  {6.0, 2.2, 4.0, 1.0},
  {6.1, 2.9, 4.7, 1.4},
  {5.6, 2.9, 3.6, 1.3},
  {6.7, 3.1, 4.4, 1.4},
  //
  {6.3, 3.3, 6.0, 2.5},
  {5.8, 2.7, 5.1, 1.9},
  {7.1, 3.0, 5.9, 2.1},
  {6.3, 2.9, 5.6, 1.8},
  {6.5, 3.0, 5.8, 2.2},
  {7.6, 3.0, 6.6, 2.1},
  {4.9, 2.5, 4.5, 1.7},
  {7.3, 2.9, 6.3, 1.8},
  {6.7, 2.5, 5.8, 1.8},
  {7.2, 3.6, 6.1, 2.5},
  {6.5, 3.2, 5.1, 2.0},
  {6.4, 2.7, 5.3, 1.9},
  {6.8, 3.0, 5.5, 2.1},
  {5.7, 2.5, 5.0, 2.0},
  {5.8, 2.8, 5.1, 2.4},
  {6.4, 3.2, 5.3, 2.3},
};

float k_means[input_data][k + 1] = {}; //distance to cluster centres

float clusters[k][attributes] = {}; //cluster center points
//######################################################################################################
int iterations = 0;
int maximum_iterations = 10;
boolean error = true;
unsigned long time;
//######################################################################################################
void setup() {

  Serial.begin(9600);
  Serial.println(F("Arduino K-Mean Clustering"));
}
//-----------------------------------------------------------------------------------------------------------------
void loop() {

  Serial.println(F("Start:"));

  while (error == true) {
    error = false;
    start_clustering(input_data, attributes, k);
  }

  runtime();

  Serial.println(F("End"));
  for (;;) {
    // endless loop
  }
}
//-----------------------------------------------------------------------------------------------------------------
void start_clustering(int input_data_set, int attribute_dat_set, int k_value) {

  Serial.println("CLUSTERS: K=" + String(k_value));

  //--------------------------------------------------------

  Serial.println(F("INITIAL CLUSTER BY RANDOM:"));

  int random_number;
  //randomSeed(analogRead(0));

  for (int i = 0; i < k_value; i++) {
    random_number = random(input_data_set - 1); //  random(0 to max)
    randomSeed(random_number);
    Serial.println("Random number:" + String(random_number));

    for (int j = 0; j < attribute_dat_set; j++) {
      clusters[i][j] = training_data_set[random_number][j];// copy k * random inputs into the cluster array
    }
  }
  show_cluster_table(k_value, attribute_dat_set);

  //--------------------------------------------------------

  boolean quality = false;//quality of clustering

  while (quality == false ) {

    quality = true;
    iterations++;
    Serial.println("Iterations:" + String(iterations));


    Serial.println(F("CALCULATE DISTANCE:")); //distance from the clusters

    float results[attribute_dat_set] = {};
    float total_result = 0 ;

    for (int k = 0; k < k_value; k++) {
      for (int i = 0; i < input_data_set; i++) {
        for (int j = 0; j < attribute_dat_set; j++) {
          results[j] = fabs(clusters[k][j] - training_data_set[i][j]);
          total_result += results[j];
        }
        k_means[i][k] = total_result;
        total_result = 0;
      }
    }

    show_cluster_table(k_value, attribute_dat_set);
    show_k_means_table(input_data_set, k_value);

    //--------------------------------------------------------

    Serial.println(F("CLASSIFICATION:")); //make clustering of all input data

    float minimum_value = 32768;
    int cluster_number;

    for (int i = 0; i < input_data_set; i++) {
      for (int k = 0; k < k_value; k++) {
        if (k_means[i][k] < minimum_value) {
          minimum_value = k_means[i][k];
          cluster_number = k;
          //Serial.println("min:" + String(minimum_value) + "," + String(k));
        }
      }
      k_means[i][k_value] = cluster_number;
      minimum_value = 32768;
      Serial.println("Data set:" + String(i) + " class=" + String(k_means[i][k_value] ));
    }


    Serial.println(F("UPDATE CENTROID:"));//update of cluster centre points

    float middle_value = 0;
    float counter = 0;

    for (int k = 0; k < k_value; k++) {
      for (int j = 0; j < attribute_dat_set; j++) {
        for (int i = 0; i < input_data_set; i++) {
          //Serial.println("index:" + String(k) + "," + String(j) +  "," + String(i) );
          if (k_means[i][k_value] == k) {
            counter++;
            middle_value += training_data_set[i][j];
            //Serial.println("found:" + String(training_data_set[i][j]) + "," + String(middle_value));
          }
        }
        //Serial.println("result:" + String(middle_value) + "/" + String(counter));
        middle_value /= counter;
        counter = 0;
        if (clusters[k][j] != middle_value) quality = false;//good results > if not longer changes of values
        clusters[k][j] = middle_value;
        middle_value = 0;
        //Serial.println("Update Clustering:" + String(k) + "," + String(clusters[k][j]));
      }
    }

    show_k_means_table(input_data_set, k_value);
    show_cluster_table(k_value, attribute_dat_set);

    if (iterations == maximum_iterations) {
      error = true;
      break;
    }
  }

  Serial.println(F("FINAL STATE:"));
  Serial.println("Iterations=" + String(iterations));
  //show_k_means_table(input_data_set, k_value);
  //show_cluster_table(k_value, attribute_dat_set);

  measure_variation(input_data_set, attribute_dat_set, k_value);
}
//-----------------------------------------------------------------------------------------------------------------
void measure_variation(int input_data_set, int attribute_dat_set, int k_value) {

  //Ziel von k-Means ist es, den Datensatz so in k Partitionen zu teilen,
  //dass die Summe der quadrierten Abweichungen von den Cluster-Schwerpunkten minimal ist.
  //Mathematisch entspricht dies der Optimierung der Funktion

  float result = 0;
  float variation[input_data_set] = {}; //variation of input data to the cluster centres

  for (int i = 0; i < input_data_set; i++) {
    for (int j = 0; j < attribute_dat_set; j++) {
      for (int k = 0; k < k_value; k++) {
        float value_1 = clusters[k][j];
        float value_2 = training_data_set[i][j];
        result += pow(value_1 - value_2, 2);
      }
    }
    result = sqrt(result);
    variation[i] = result;
    result = 0;
  }

  //--------------------------------

  float tolal_variation = 0;
  for (int i = 0; i < input_data_set; i++) {
    tolal_variation += variation[i];
  }
  Serial.println("Total Variation:" + String(tolal_variation) + "  K=" + String(k_value)); //you need an elbow plot to check the best k-value by variaton
  calculate_gradient(k_value, tolal_variation);
}
//-----------------------------------------------------------------------------------------------------------------
void calculate_gradient(float a, float b) {

  float c = sqrt((a * a) + (b * b));
  float alfa = acos(((a * a) + (c * c) - (b * b)) / (2 * a * c));
  alfa = alfa * 180 / 3.14159;//rad to deg
  Serial.println("Gradient: alfa=" + String(alfa, 5) + " Deg");
}
//-----------------------------------------------------------------------------------------------------------------
void show_k_means_table(int input_data_set, int k_value) {

  Serial.println("Show k-means table:");
  for (int z = 0; z < input_data_set; z++) {
    Serial.print("Show distance to cluster centre:" + String(z) + ";");
    for (int k = 0; k < k_value + 1; k++) {
      Serial.print(String(k_means[z][k]) + "," );
    }
    Serial.println();
  }
  Serial.println("------");
}
//-----------------------------------------------------------------------------------------------------------------
void show_cluster_table(int k_value, int attribute_dat_set) {

  Serial.println("Show cluster table:");
  for (int k = 0; k < k_value; k++) {
    Serial.print("Show cluster centres:" + String(k) + ";");
    for (int j = 0; j < attribute_dat_set; j++) {
      Serial.print(String(clusters[k][j]) + ",");
    }
    Serial.println();
  }
  Serial.println("------");
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


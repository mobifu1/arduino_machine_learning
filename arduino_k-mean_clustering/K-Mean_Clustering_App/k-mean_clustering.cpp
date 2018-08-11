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

#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

using namespace std;

#define input_data 48
#define attributes 4
const int  k = 3; //number of clusters to divide the input data 
/*
                                                                                Elbow-Plot:  | \
Elbow-Plot X,Y:                                                                              |  \
K=0 > Variance=NAN                                                                           |   \
K=1 > Variance=96.39                                                                         |    \
K=2 > Variance=43.61    //optimum!!!                                                         |     \
K=3 > Variance=33.48                                                                         |      --------------
K=4 > Variance=28.46                                                                         |___________________________K
K=5 > Variance=25.61
//-------------------------------------------------------------------------------------------------------------------------------------------------
*/
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
bool error = true;
//######################################################################################################
float create_random() { // random numbes beetwen 0 and length of input data

	float randNum = rand() % input_data;
	return randNum;
}
//-----------------------------------------------------------------------------------------------------------------
void show_k_means_table(int input_data_set, int k_value) {

  cout << "Show k-means table:" << endl;
  for (int z = 0; z < input_data_set; z++) {
    cout << "Show distance to all cluster centre:" << z << ";";
    for (int k = 0; k < k_value + 1; k++) {
      cout << k_means[z][k] << ",";
    }
    cout << endl;
  }
  cout << "------" << endl;
}
//-----------------------------------------------------------------------------------------------------------------
void show_cluster_table(int k_value, int attribute_dat_set) {

  cout << "Show cluster table:" << endl;
  for (int k = 0; k < k_value; k++) {
    cout << "Show cluster centres:" << k << ";";
    for (int j = 0; j < attribute_dat_set; j++) {
      cout << clusters[k][j] << ",";
    }
    cout << endl;
  }
  cout << "------" << endl;
}
//------------------------------------------------------------------------------------------------------------------
void measure_variance(int input_data_set, int attribute_dat_set, int k_value) {

  //Ziel von k-Means ist es, den Datensatz so in k Partitionen zu teilen,
  //dass die Summe der quadrierten Abweichungen von den Cluster-Schwerpunkten minimal ist.
  //Mathematisch entspricht dies der Optimierung der Funktion

  float result = 0;
  float variation[input_data_set] = {}; //variation of input data to the alocated cluster centre

  cout << "Measure variance" << endl;
  for (int i = 0; i < input_data_set; i++) {
    int value_1 = int(k_means[i][k_value]); //class
    //cout << "Show distance to alocated cluster centre:" << i << ";";
    for (int j = 0; j < attribute_dat_set; j++) {
      float value_2 = clusters[value_1][j];
      float value_3 = training_data_set[i][j];
      result += pow(value_3 - value_2, 2);
    }
    result = sqrt(result);
    variation[i] = result;
    result = 0;
    //cout << variation[i];
    //cout << endl;
  }

  //--------------------------------

  float tolal_variation = 0;
  for (int i = 0; i < input_data_set; i++) {
    tolal_variation += variation[i];
  }
  cout << "Total variance:" << tolal_variation << "  K=" << k_value << endl; //you need an elbow plot to check the best k-value by variaton
}
//-----------------------------------------------------------------------------------------------------------------
void start_clustering(int input_data_set, int attribute_data_set, int k_value) {

  cout << "CLUSTERS: K=" << k_value << endl;

  //--------------------------------------------------------

  cout << "INITIAL CLUSTER BY RANDOM:" << endl;

  int random_number;

  for (int i = 0; i < k_value; i++) {
    random_number = create_random(); //  random(0 to max)
    cout << "Random number:" << random_number << endl;

    for (int j = 0; j < attribute_data_set; j++) {
	   clusters[i][j] = training_data_set[random_number][j];// copy k * random inputs into the cluster array
    }
  }
  show_cluster_table(k_value, attribute_data_set);

  //--------------------------------------------------------

  bool quality = false;//quality of clustering

  while (quality == false ) {

    quality = true;
    iterations++;
    cout << "Iterations:" << iterations << endl;


    cout << "CALCULATE DISTANCE:" << endl; //distance from the clusters

    float results[attribute_data_set] = {};
    float total_result = 0 ;

    for (int k = 0; k < k_value; k++) {
      for (int i = 0; i < input_data_set; i++) {
        for (int j = 0; j < attribute_data_set; j++) {
          results[j] = fabs(clusters[k][j] - training_data_set[i][j]);
          total_result += results[j];
        }
        k_means[i][k] = total_result;
        total_result = 0;
      }
    }

    show_k_means_table(input_data_set, k_value);
    show_cluster_table(k_value, attribute_data_set);

    //--------------------------------------------------------

    cout << "CLASSIFICATION:" << endl; //make clustering of all input data

    float minimum_value = 32768;
    int cluster_number;

    for (int i = 0; i < input_data_set; i++) {
      for (int k = 0; k < k_value; k++) {
        if (k_means[i][k] < minimum_value) {
          minimum_value = k_means[i][k];
          cluster_number = k;
          //cout << "min:" << minimum_value << "," << k << endl;
        }
      }
      k_means[i][k_value] = cluster_number;
      minimum_value = 32768;
      cout << "Data set:" << i << " class=" << k_means[i][k_value] << endl;
    }

    cout << "UPDATE CENTROID:" << endl;//update of cluster centre points

    float middle_value = 0;
    float counter = 0;

    for (int k = 0; k < k_value; k++) {
      for (int j = 0; j < attribute_data_set; j++) {
        for (int i = 0; i < input_data_set; i++) {
          //cout << "index:" << k << "," << j << "," << i <<endl;
          if (k_means[i][k_value] == k) {
            counter++;
            middle_value += training_data_set[i][j];
            //cout << "found:" << training_data_set[i][j] << "," << middle_value << endl;
          }
        }
        //cout << "result:" << middle_value << "/" << counter << endl;
        middle_value /= counter;
        counter = 0;
        if (clusters[k][j] != middle_value) quality = false;//good results > if not longer changes of values
        clusters[k][j] = middle_value;
        middle_value = 0;
        //cout << "Update Clustering:" << k << "," << clusters[k][j] <<endl;
      }
    }

    show_k_means_table(input_data_set, k_value);
    show_cluster_table(k_value, attribute_data_set);

    if (iterations == maximum_iterations) {
      error = true;
      break;
    }
  }

  cout << "FINAL STATE:" << endl;
  cout << "Iterations=" << iterations << endl;
  //show_k_means_table(input_data_set, k_value);
  //show_cluster_table(k_value, attribute_dat_set);

  measure_variance(input_data_set, attribute_data_set, k_value);
}
//######################################################################################################
int main() {

  cout << "Start:" << endl;

  while (error == true) {
    error = false;
    start_clustering(input_data, attributes, k);
  }

  cout << "End" << endl;
  for (;;) {
    // endless loop
  }
  	return 0;
}
//-----------------------------------------------------------------------------------------------------------------


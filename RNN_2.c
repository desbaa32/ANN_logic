#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Taille des couches
#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define INPUT_SIZE_LOGIC 2
#define OUTPUT_SIZE_LOGIC 4

// Structure réseau de neurones
typedef struct {
    double input[INPUT_SIZE];
    double hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    double hidden_bias[HIDDEN_SIZE];
    double hidden_layer[HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE];
    double output_bias;
    double output_layer;
    double learning_rate;
} NeuralNetwork;
typedef struct {
    double inputs[OUTPUT_SIZE_LOGIC][INPUT_SIZE_LOGIC];
    double targets[OUTPUT_SIZE_LOGIC];
} AND_logic;
// Fonction d'initialisation des poids et biais [valeurs aléatoires] du réseau de neurones
void initialize(NeuralNetwork *network) {
    // Initialisation des poids cachés
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            network->hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    // Initialisation des biais cachés
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        network->hidden_bias[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Initialisation des poids de sortie
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        network->output_weights[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Initialisation du biais de sortie
    network->output_bias = ((double)rand() / RAND_MAX) * 2 - 1;
}

// Fonction d'activation
double tanh_activation(double x) {
  // return 1.0 / (1.0 + exp(-x));
    return tanh(x);
}

// Fonction de dérivée de la fonction d'activation tangente hyperbolique. 
double tanh_activation_derivative(double x) {
    //return x * (1.0 - x);
    return 1.0 - tanh(x) * tanh(x);
}

// Fonction de propagation avant
void forwardPropagation(NeuralNetwork *network) {
    // Couche cachée
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        double sum = 0.0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            sum += network->input[i] * network->hidden_weights[i][j];
        }
        network->hidden_layer[j] = tanh_activation(sum + network->hidden_bias[j]);
    }

    // Couche de sortie
    double sum_output = 0.0;
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        sum_output += network->hidden_layer[j] * network->output_weights[j];
    }
    network->output_layer = tanh_activation(sum_output + network->output_bias);
}

// Fonction de rétropropagation
void backPropagation(NeuralNetwork *network, double target) {
    // Calcul de l'erreur de sortie
    double output_error = target - network->output_layer;

    // Calcul du gradient de la couche de sortie
    double output_delta = output_error * tanh_activation_derivative(network->output_layer);

    // Mise à jour des poids de la couche de sortie
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        network->output_weights[j] += network->learning_rate * output_delta * network->hidden_layer[j];
    }

    // Mise à jour du biais de la couche de sortie
    network->output_bias += network->learning_rate * output_delta;

    // Calcul de l'erreur de la couche cachée
    double hidden_error[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        hidden_error[j] = output_delta * network->output_weights[j];
    }

    // Calcul du gradient de la couche cachée
    double hidden_delta[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        hidden_delta[j] = hidden_error[j] * tanh_activation_derivative(network->hidden_layer[j]);
    }

    // Mise à jour des poids de la couche cachée
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            network->hidden_weights[i][j] += network->learning_rate * hidden_delta[j] * network->input[i];
        }
    }

    // Mise à jour des biais de la couche cachée
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        network->hidden_bias[j] += network->learning_rate * hidden_delta[j];
    }
}

// Fonction d'entraînement
void train(NeuralNetwork *network, double target, int iteration) {

  for (int i = 0; i < iteration; i++) {
    forwardPropagation(network);
    backPropagation(network,target);
  }}
  void train_logic(NeuralNetwork *network,  AND_logic *logict, int num_iterations) {
  // Boucle d'entraînement
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Présenter chaque exemple d'entraînement
        for (int logic = 0; logic < 4; ++logic) {
            // Définir les entrées et la sortie cible pour cet exemple
            for (int i = 0; i < INPUT_SIZE; ++i) {
                network->input[i] = logict->inputs[logic][i];
            }
            double target = logict->targets[logic];

            // Effectuer la propagation avant
            forwardPropagation(network);

            // Effectuer la rétropropagation
            backPropagation(network, target);
        }
    }
    // Tester le réseau après l'entraînement
    for (int logic = 0; logic < 4; ++logic) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            network->input[i] = logict->inputs[logic][i];
        }

        // Effectuer la propagation avant
        forwardPropagation(network);

        // Afficher la sortie après l'entraînement
        printf("Output after training for example %d: %lf\n",logic, network->output_layer);
         //printf("for example [%d \n",logict->inputs[0][logic]);
        // printf("Output after training for example [%d - %d]: %lf\n", and_logic->inputs[logic][0],and_logic->inputs[logic][1], network->output_layer);
    }
  
  }
 void Default_execution_NN(NeuralNetwork *network,double learning_rate ){
    initialize(network);
    network->learning_rate = learning_rate;
    // Entrées d'exemple
    network->input[0] = 1;
    network->input[1] = 1;
    // Sortie cible
    double target = 1;
    // Effectuer la propagation avant
    forwardPropagation(network);
    // Afficher la sortie avant la rétropropagation
    printf("Output before backpropagation: %lf\n", network->output_layer);
    // Effectuer la rétropropagation
    backPropagation(network, target);
    // Afficher la sortie après la rétropropagation
    forwardPropagation(network);
    printf("Output after backpropagation : %lf\n", network->output_layer);
 }
int main() {
    // Initialiser le réseau de neurones
    NeuralNetwork network;
    NeuralNetwork network2;
    AND_logic  and_logic = {
        .inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        },
        .targets = {0, 0, 0, 1}
    };
    AND_logic  or_logic = {
        .inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        },
        .targets = {0, 1, 1, 1}
    };
    int num_iterations = 1500000;
    double learning_rate = 0.6;
    
    printf("\t -------------- Before Trainning  ------------- \n");
    printf("\t -- AND  for example [1 - 1] -- \n");
    Default_execution_NN(&network,learning_rate);
    printf("\t -- OR  for example [1 - 1] -- \n");
    Default_execution_NN(&network2,learning_rate);
    printf("\t -------------- Trainning  ------------- \n");
//     train(&network,target,num_iterations);
//    // test
//     forwardPropagation(&network);
//     //printf("Prédiction :");
//     printf("Output after training : %lf\n", network.output_layer);
 printf("\t -- AND  -- \n");
    train_logic(&network, &and_logic,num_iterations);
    printf("\t -- OR  -- \n");
    train_logic(&network2, &or_logic,num_iterations+7000000);
    return 0;
}




////gcc -Wall RNN_2.c -o RNN_2 -lm && ./RNN_2
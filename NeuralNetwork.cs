namespace datasetRA
{
    public partial class NeuralNetwork
    {
        Neuron[] layer1;
        Neuron[] layer2;


        public NeuralNetwork(int pNumberInputs, int pNumberNeuronsHidden, int pNumberOutputs)
        {
            var rnd = new Random();

            layer1 = new Neuron[pNumberNeuronsHidden];
            layer2 = new Neuron[pNumberOutputs];

            for(int i = 0; i < pNumberNeuronsHidden; i++)
                layer1[i] = Neuron.Create(pNumberInputs, rnd);

            for (int i = 0; i < pNumberOutputs; i++)
                layer2[i] = Neuron.Create(pNumberNeuronsHidden, rnd);

        }

        public float[] Calculate(float[] pInputs, ref float[] pOutputs, ref float[] pHiddenOutputs)
        {
            float[] layer1_Axis = new float[layer1.Length];
            float[] layer2_Axis = new float[layer2.Length];

            for (int i = 0; i < layer1.Length; i++)
                layer1_Axis[i] = layer1[i].Calculate(pInputs);

            for (int i = 0; i < layer2.Length; i++)
                layer2_Axis[i] = layer2[i].Calculate(layer1_Axis);

            pOutputs = layer2_Axis;
            pHiddenOutputs = layer1_Axis;

            return layer2_Axis;
        }

        public float[] Calculate(float[] pInputs)
        {
            float[] layer1_Axis = new float[layer1.Length];
            float[] layer2_Axis = new float[layer2.Length];

            for (int i = 0; i < layer1.Length; i++)
                layer1_Axis[i] = layer1[i].Calculate(pInputs);

            for (int i = 0; i < layer2.Length; i++)
                layer2_Axis[i] = layer2[i].Calculate(layer1_Axis);


            return layer2_Axis;
        }


        public void Backpropagation(float[] Tinputs, float[] T, ref float[] TAxis1, ref float[] TAxis2, ref float[] Error, float learningCoefficient)
		{
            var Number_Hidden2 = layer1.Length;
            var Number_outputs = layer2.Length;
            var Number_inputs = Tinputs.Length;


            float[] deltaJ = new float[Number_Hidden2];
            float[] deltaK = new float[Number_outputs];

            float[] prodTempJ = new float[Number_Hidden2];

            Error = new float[Number_outputs];

            float sumj = 0;

			//	Processes
			for (int i = 0; i < Number_Hidden2; i++) prodTempJ[i] = 0;


			for (int i = 0; i < Number_outputs; i++) Error[i] = T[i] - TAxis2[i]; //	Error normal
																					//for (int i = 0; i < Number_outputs; i++) Error[i] = Math.Pow(T[i] - Toutputs[i],2);	//	Error Cuadrático medio

			for (int i = 0; i < Number_outputs; i++) deltaK[i] = (TAxis2[i] * (1 - TAxis2[i])) * Error[i];


			//---
			for (int j = 0; j < Number_Hidden2; j++)
			{
				for (int k = 0; k < Number_outputs; k++)
				{
					prodTempJ[j] += layer2[k].W[j] * deltaK[k];
				}
			}
			for (int i = 0; i < Number_Hidden2; i++) sumj += prodTempJ[i];
			for (int i = 0; i < Number_Hidden2; i++) deltaJ[i] = sumj * (TAxis1[i] * (1 - TAxis1[i]));



			//---
			//	Update Knowledge

			//	Layer Hidden (2)
			for (int i = 0; i < Number_Hidden2; i++)
			{
				float AlphaDelta = learningCoefficient * deltaJ[i];
				for (int j = 0; j < Number_inputs; j++)
				{
                    layer1[i].W[j] += AlphaDelta * Tinputs[j];
				}
                layer1[i].B += AlphaDelta;
			}

			//	Layer Output (3)
			for (int i = 0; i < Number_outputs; i++)
			{
				float AlphaDelta = learningCoefficient * deltaK[i];
				for (int j = 0; j < Number_Hidden2; j++)
				{
                    layer2[i].W[j] += AlphaDelta * TAxis1[j];
				}
                layer2[i].B += AlphaDelta;

            }

        }


        public void Train(float[][] P, float[][] T, int MAX_EPOCHS = 100000, float learningCoefficient = 0.05f)
        {
            float[] buffer_layer1 = new float[layer1.Length];
            float[] buffer_layer2 = new float[layer2.Length];
            float[][] buffer_error = new float[4][];

            int epochs;

            for (epochs = 0; epochs < MAX_EPOCHS; epochs++)
            {
                for (int j = 0; j < T.Length; j++)
                {
                    Calculate(P[j], ref buffer_layer2, ref buffer_layer1);
                    Backpropagation(P[j], T[j], ref buffer_layer1, ref buffer_layer2, ref buffer_error[j], learningCoefficient);
                }

                if (epochs % 1000 == 0)
                {
                    var meanSquaredError = buffer_error.Select(e_vector => e_vector.Select(e => Math.Pow(e, 2)).Sum()).Sum() / T.Length;

                    Console.WriteLine("Mean squared error:\t" + meanSquaredError);

                    if (meanSquaredError < 0.01)
                        break;
                }
            }

            Console.WriteLine("Training finished in " + epochs + " epochs.");
        }



        public void BackPropagation_deprecated(
            float[] pOutputs,
            float[] pOutputsHidden,
            float[] pT,
            float[] pP,
            float pLearningRate = 0.05f)
        {
            //  Output layer
            float[] eOutput = new float[layer2.Length];
            float[] eHidden = new float[layer1.Length];
            float[] deltaOutput = new float[layer2.Length];
            float[] deltaHidden = new float[layer1.Length];

            for (int i = 0; i < layer2.Length; i++)
            {
                //  Calculate error
                eOutput[i] = pOutputs[i] - pT[i]; //  TODO: Evaluate expression
                //  Derived Sigmoid * error
                deltaOutput[i] = pOutputs[i] * (1 - pOutputs[i]) * eOutput[i];
            }

            for (int i = 0; i < layer1.Length; i++)
            {
                //eHidden[i] = Dot(new float[][] { deltaOutput }, new float[][] { layer2[i].W });
                deltaHidden[i] = eHidden[i] * pOutputsHidden[i] * pOutputsHidden[i] * (1 - pOutputsHidden[i]);
            }



            for (int i = 0; i < layer2.Length; i++)
            {
                for (int j = 0; j < layer2[i].W.Length; j++)
                    layer2[i].W[j] -= pLearningRate * deltaOutput[i] * pOutputsHidden[j];

                layer2[i].B -= pLearningRate * deltaOutput[i];
            }

            for (int i = 0; i < layer1.Length; i++)
            {
                for (int j = 0; j < layer1[i].W.Length; j++)
                    layer1[i].W[j] -= pLearningRate * deltaHidden[i] * pP[j];

                layer2[i].B -= pLearningRate * deltaHidden[i];
            }


        }

        partial class Neuron
        {
            public float B { get; set; }
            public float[] W { get; set; }
            public int InputLength { get; private set; }

            public Neuron()
            {
                InputLength = 0;
                W = new float[0];
            }

            public float Calculate(float[] pInputs)
            {
                if (InputLength == 0) 
                    throw new Exception("Neuron no instanciado.");

                var productScalar = B;

                for(int i = 0; i < InputLength; i++)
                    productScalar += pInputs[i] * W[i];

                return Sigmoid(productScalar);
            }


           


            public static Neuron Create(int pInputLength, Random pRndObj)
            {
                Neuron oNeuron = new();

                oNeuron.W = new float[pInputLength];
                oNeuron.B = pRndObj.Next(-200, 200) / 100f;
                oNeuron.InputLength = pInputLength;

                for (int i = 0; i < pInputLength; i++)
                    oNeuron.W[i] = pRndObj.Next(-200, 200) / 100f;

                return oNeuron;
            }
        }



        private static float Sigmoid(float x)
        {
            return (float)(1 / (1 + (Math.Pow(Math.E, -x))));
        }

    }
}

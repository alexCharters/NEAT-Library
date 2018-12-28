using System;
using System.Collections.Generic;
using System.IO;
using NEAT;

namespace MAIN
{
	class MAIN
	{
		static List<List<double>> inputsList;
		static List<int> outputs;
		static void Main(string[] args)
		{
			using (StreamReader reader = new StreamReader("C:/Users/Alex/Desktop/creditcard.csv"))
			{
				inputsList = new List<List<double>>();
				outputs = new List<int>();
				reader.ReadLine();
				int count = 0;
				while (!reader.EndOfStream)
				{

					var line = reader.ReadLine();
					var values = line.Split(',');

					inputsList.Add(new List<double>());
					for (int i = 0; i < 30; i++)
					{
						inputsList[count].Add(double.Parse(values[i]));
					}
					outputs.Add(Int32.Parse(values[30].Trim(new char[] { '"', '\\' })));
					count++;
				}
			}

			Population pop = new Population(30, new List<string>() { "fraud" }, 200, FitnessFunction);

			while (true)
			{
				pop.Run();
				pop.Select();

				Console.WriteLine(pop.AvgPopFitness);
			}

			//NeuralNetwork NN = new NeuralNetwork(inputs, outputs, new Dictionary<int, Tuple<int, int>>());
			//for (int i = 0; i < 1000; i++)
			//{
			//	NN.MutateLink();
			//}

			//for (int i = 0; i < 1000; i++)
			//{
			//	NN.MutateNeuron();
			//}

			//Dictionary<Tuple<int, int>, int> innovationNumbers = new Dictionary<Tuple<int, int>, int>();

			//NeuralNetwork NN1 = new NeuralNetwork(inputs, outputs, innovationNumbers);
			//NeuralNetwork NN2 = new NeuralNetwork(inputs, outputs, innovationNumbers);
			//for (int i = 0; i < 15; i++)
			//{
			//	NN1.MutateLink();
			//	NN2.MutateLink();
			//}

			//NN2.MutateNeuron();
			//NN2.MutateNeuron();
			//NN2.MutateNeuron();
			//NN2.MutateNeuron();

			//for (int i = 0; i < 15; i++)
			//{
			//	NN2.MutateLink();
			//}

			//NN1.InitializeRandom();
			//NN2.InitializeRandom();
			//NN1.Fitness = 100;
			//NN2.Fitness = 90;

			//Console.WriteLine(NN1.GenerateDOT());
			//Console.WriteLine(NN2.GenerateDOT());

			//Console.WriteLine();
			//Console.WriteLine();
			//Console.WriteLine(NeuralNetwork.EvolutionaryDistance(NN1, NN2, innovationNumbers));
			//Console.WriteLine();
			//Console.WriteLine();

			//NeuralNetwork child = NeuralNetwork.Crossover(NN1, NN2, innovationNumbers);
			//Console.WriteLine(child.GenerateDOT());

			//Console.WriteLine("Done");
			//Console.WriteLine();
			//Console.WriteLine();
			//Population pop = new Population(GetInputs(), GetOutputs(), 200, RunFunction);
			//pop.Run();
			//pop.Select();
			//foreach (Species species in pop.speciesList)
			//{
			//	Console.WriteLine("SPECIES:");
			//	foreach (NeuralNetwork NN in species.networks)
			//	{
			//		Console.WriteLine(NN);
			//		Console.WriteLine();
			//	}
			//	Console.WriteLine("Adjusted Species Fitness Sum: "+species.adjustedFitnessSum);
			//	Console.WriteLine();
			//	Console.WriteLine();
			//}
			//Console.ReadLine();
		}

		public static List<double> GetInputs()
		{
			return new List<double>
			{
				12,
				1,
				2,
				7,
				8,
				122,
				69
			};
		}

		public static List<string> GetOutputs()
		{
			return new List<string>
			{
				"Out",
				"Also Out",
				"yeeeet"
			};
		}

		public static Random rand = new Random();
		public static double FitnessFunction(NeuralNetwork NN)
		{
			double avgPerformance = 0;
			for (int i = 0; i < inputsList.Count; i++)
			{
				List<double> inputs = inputsList[i];
				NN.Evaluate(inputs).TryGetValue("fraud", out double result);
				if ((result > .5 && outputs[i] == 1) || (result < .5 && outputs[i] == 0))
				{
					avgPerformance += 1;
				}
			}
			return avgPerformance / inputsList.Count;
		}
	}
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NEAT
{
	/// <summary>
	/// This class represents a population, a collection of neural networks. The Neural Networks can be repeatedly evaluated, selected, and reproduced to evolve specialized Neural Networks.
	/// </summary>
	public class Population
	{
		readonly Dictionary<Tuple<int, int>, int> innovationNumbers;
		public delegate double analysisFunction(NeuralNetwork NN);
		private readonly analysisFunction run;
		public List<NeuralNetwork> Networks { get; private set; }
		public List<Species> speciesList;
		public double AvgPopFitness { get; private set; }
		public int Size { get; private set; }
		public NeuralNetwork BestNetwork { get; private set; }

		public bool threaded = false;
		public double SpeciatingThreshold = 3;
		public double killRate = .9;

		/// <summary>
		/// Constructor for a population. Creates the first generation of Neural Networks.
		/// </summary>
		/// <param name="numInputs">The number of inputs each Network has.</param>
		/// <param name="outputNames">An Enumerable of the names of each output.</param>
		/// <param name="PopulationSize">The size of the generated populations</param>
		/// <param name="_run">The Fitness Function.</param>
		public Population(int numInputs, IEnumerable<string> outputNames, int PopulationSize, analysisFunction _run)
		{
			run = _run;
			innovationNumbers = new Dictionary<Tuple<int, int>, int>();
			Networks = new List<NeuralNetwork>();
			speciesList = new List<Species>();
			Size = PopulationSize;
			for (int i = 0; i < PopulationSize; i++)
			{
				NeuralNetwork NN = new NeuralNetwork(numInputs, outputNames, innovationNumbers);
				NN.InitializeRandom();
				Networks.Add(NN);
			}
		}

		/// <summary>
		/// Checks all Networks list of connections and ensure the phenotype reflects it.
		/// </summary>
		/// <returns>true if all networks are consistent</returns>
		public bool IsConsistent()
		{
			bool consistent = true;
			foreach (NeuralNetwork NN in Networks) {
				if (!NN.IsConsistent()) {
					consistent = false;
				}
			}
			return consistent;
		}

		/// <summary>
		/// Runs each Neural Network of the population through the fitness function and assigns the fitnesses.
		/// Overloaded for running on multiple threads. default is unthreaded.
		/// </summary>
		public void Run()
		{
			Dictionary<int, double> fitnesses = new Dictionary<int, double>();
			double totalfitness = 0;

			if (threaded)
			{
				Thread thread1 = new Thread(() =>
				{
					for (int i = 0; i < Networks.Count / 4; i++)
					{
						NeuralNetwork NN = Networks.ElementAt(i);
						double fitness = run(NN);
						totalfitness += fitness;
						lock (fitnesses)
						{
							fitnesses.Add(i, fitness);
						}
						//Console.WriteLine(i + " -> " + fitness);
					}
				});
				Thread thread2 = new Thread(() =>
				{
					for (int i = Networks.Count / 4; i < Networks.Count / 2; i++)
					{
						NeuralNetwork NN = Networks.ElementAt(i);
						double fitness = run(NN);
						totalfitness += fitness;
						lock (fitnesses)
						{
							fitnesses.Add(i, fitness);
						}
						//Console.WriteLine(i + " -> " + fitness);
					}
				});
				Thread thread3 = new Thread(() =>
				{
					for (int i = Networks.Count / 2; i < 3 * Networks.Count / 4; i++)
					{
						NeuralNetwork NN = Networks.ElementAt(i);
						double fitness = run(NN);
						totalfitness += fitness;
						lock (fitnesses)
						{
							fitnesses.Add(i, fitness);
						}
						//Console.WriteLine(i + " -> " + fitness);
					}
				});
				Thread thread4 = new Thread(() =>
				{
					for (int i = 3 * Networks.Count / 4; i < Networks.Count; i++)
					{
						NeuralNetwork NN = Networks.ElementAt(i);
						double fitness = run(NN);
						totalfitness += fitness;
						lock (fitnesses)
						{
							fitnesses.Add(i, fitness);
						}
						//Console.WriteLine(i + " -> " + fitness);
					}
				});

				thread1.Start();
				thread2.Start();
				thread3.Start();
				thread4.Start();

				thread1.Join();
				thread2.Join();
				thread3.Join();
				thread4.Join();
			}
			else
			{
				int count = 0;
				foreach (NeuralNetwork NN in Networks)
				{
					double fitness = run(NN);
					totalfitness += fitness;
					fitnesses.Add(count, fitness);
					//Console.WriteLine(count + " -> " + fitness);
					count++;
				}
			}

			AvgPopFitness = totalfitness / Networks.Count;
			BestNetwork = null;

			for (int i = 0; i < Networks.Count(); i++)
			{
				fitnesses.TryGetValue(i, out double fitness);
				if (BestNetwork == null || fitness > BestNetwork.Fitness) {
					BestNetwork = Networks.ElementAt(i);
				}
				Networks.ElementAt(i).Fitness = fitness;
			}
			//Console.WriteLine("Run: " + isConsistent());
		}

		/// <summary>
		/// Runs each Neural Network of the population through the fitness function and assigns the fitnesses.
		/// </summary>
		/// <param name="_threaded">true for multithreading</param>
		public void Run(bool _threaded)
		{
			this.threaded = _threaded;
			Run();
		}

		/// <summary>
		/// Performs speciation, adjusting fitness sums, culling, crossover, and mutation, to generate the next generation of Neural Networks
		/// </summary>
		public void Select()
		{
			speciesList = new List<Species>();
			foreach (NeuralNetwork NN in Networks)
			{
				bool foundSpecies = false;
				foreach (Species species in speciesList)
				{
					if (NeuralNetwork.EvolutionaryDistance(species.networks.ElementAt(0), NN, innovationNumbers) < SpeciatingThreshold)
					{
						species.networks.Add(NN);
						foundSpecies = true;
						break;
					}
				}
				if (!foundSpecies)
				{
					speciesList.Add(new Species(NN));
				}
			}

			//Console.WriteLine("speciesList Count after speciating: " + speciesList.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());

			foreach (Species species in speciesList)
			{
				species.networks.Sort((NN1, NN2) => NN2.Fitness.CompareTo(NN1.Fitness));
				int kill = (int)(Math.Ceiling(species.networks.Count() * killRate));
				int total = species.networks.Count();
				for (int i = species.networks.Count() - 1; i > total - 1 - kill; i--)
				{
					species.networks.RemoveAt(i);
				}

				for (int i = 0; i < species.networks.Count(); i++)
				{
					species.adjustedFitnessSum += species.networks.ElementAt(i).Fitness / (species.networks.Count);
				}
			}

			speciesList.RemoveAll(x => x.networks.Count < 3);

			//Console.WriteLine("speciesList Count after killing: " + speciesList.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());

			int numSelectionBreed = (int)(.75 * Size);

			speciesList.Sort((species1, species2) => species2.adjustedFitnessSum.CompareTo(species1.adjustedFitnessSum));

			double sharedFitnessTotal = 0;
			for (int i = 0; i < speciesList.Count / 3; i++)
			{
				Species species = speciesList.ElementAt(i);
				sharedFitnessTotal += species.adjustedFitnessSum;
			}

			//Console.WriteLine("speciesList Count after adjusting fitness sums: " + speciesList.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());

			List<NeuralNetwork> childrenNetworks = new List<NeuralNetwork>();
			Random rand = new Random();
			for (int i = 0; i < speciesList.Count / 3; i++)
			{
				Species species = speciesList.ElementAt(i);
				for (int j = 0; j < numSelectionBreed * (species.adjustedFitnessSum / sharedFitnessTotal); j++)
				{
					//Console.WriteLine("Pop Consistency: " + isConsistent());
					NeuralNetwork NN1 = species.networks.ElementAt(rand.Next(species.networks.Count));
					NeuralNetwork NN2 = species.networks.ElementAt(rand.Next(species.networks.Count));
					NeuralNetwork Child = NeuralNetwork.Crossover(NN1, NN2, innovationNumbers);
					Child.RandomMutation();
					//Child.DisplayOutputConnections();
					childrenNetworks.Add(Child);
					//Console.WriteLine("Pop Consistency: " + isConsistent());
					//Console.WriteLine();
				}
			}

			//Console.WriteLine();
			//Console.WriteLine("speciesList Count after selection breeding: "+ speciesList.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());
			//Console.WriteLine();

			Networks.Sort((NN1, NN2) => NN2.Fitness.CompareTo(NN1.Fitness));
			for (int i = 0; i < 5; i++)
			{
				childrenNetworks.Add(Networks.ElementAt(i));
			}

				int numRandomBreed = Networks.Count - childrenNetworks.Count;
			for (int i = 0; i < numRandomBreed; i++)
			{
				Species randSpecies = speciesList[rand.Next(speciesList.Count)];

				NeuralNetwork randNN1 = randSpecies.networks[rand.Next(randSpecies.networks.Count)];
				NeuralNetwork randNN2 = randSpecies.networks[rand.Next(randSpecies.networks.Count)];
				NeuralNetwork child = NeuralNetwork.Crossover(randNN1, randNN2, innovationNumbers);
				child.RandomMutation();
				//child.DisplayOutputConnections();
				childrenNetworks.Add(child);
			}

			//Console.WriteLine();
			//Console.WriteLine("speciesList Count after random breeding: " + speciesList.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());
			//Console.WriteLine();

			Networks = new List<NeuralNetwork>(childrenNetworks);

			//Console.WriteLine("total child networks after selection: " + childrenNetworks.Count);
			//Console.WriteLine("Pop Consistency: " + isConsistent());
		}
	}

	/// <summary>
	/// A species is a group of Neural Networks and an associated adjusted fitness sum.
	/// </summary>
	public class Species
	{
		public double adjustedFitnessSum;
		public List<NeuralNetwork> networks;

		/// <summary>
		/// instantiates a new Species.
		/// </summary>
		/// <param name="NN">The first NN of the species. Used for all evolutionary distance calculations for this species</param>
		public Species(NeuralNetwork NN)
		{
			networks = new List<NeuralNetwork>() { NN };
		}
	}

	/// <summary>
	/// Represents an organism within the population. An acyclic graph consisting of neurons and connections.
	/// </summary>
	public class NeuralNetwork
	{
		public Dictionary<int, Neuron> Neurons { get; private set; }
		public List<Tuple<int, int>> Connections { get; private set; }
		public int NumOutputs { get; private set; }
		public int NumInputs { get; private set; }
		public int NumNeurons { get; private set; }
		public Dictionary<Tuple<int, int>, int> InnovationNumbers { get; private set; }
		public double Fitness { get; set; }
		List<string> outputNames;

		public static double disjointCoef = 1.0;
		public static double excessCoef = 1.0;
		public static double avgWeightDiffCoef = 0.4;

		public double connectionWeightsMutationChance = .8;
		public double connectionMutationChance = .1;
		public double NeuronMutationChance = .06;

		Random rand = new Random(Guid.NewGuid().GetHashCode());

		/// <summary>
		/// Constructs a NN based off of the number of inouts, the number and names of all outputs.
		/// </summary>
		/// <param name="numInputs">The number of inputs this neural network has.</param>
		/// <param name="outputNames">Enumerable of the names of the outputs.</param>
		/// <param name="_innovationNumbers">innovation numbers of the entire population</param>
		public NeuralNetwork(int numInputs, IEnumerable<string> outputNames, Dictionary<Tuple<int, int>, int> _innovationNumbers)
		{
			this.outputNames = (List<string>)outputNames;
			Neurons = new Dictionary<int, Neuron>();
			Connections = new List<Tuple<int, int>>();
			InnovationNumbers = _innovationNumbers;
			NumInputs = numInputs+1;
			NumOutputs = outputNames.Count();
			NumNeurons = NumInputs + NumOutputs;
			Fitness = double.MinValue;
			for (int id = 0; id < NumInputs; id++)
			{
				Neurons.Add(id, new Neuron(id));
			}

			Neurons[0].Value = 1;

			for (int id = 0; id < NumOutputs; id++)
			{
				Neurons.Add(id + NumInputs, new Neuron(id + NumInputs));
			}
		}

		/// <summary>
		/// Makes the initial connections frome every inout to every output with random weights.
		/// </summary>
		public void InitializeRandom()
		{
			for (int i = NumInputs; i < NumInputs+NumOutputs; i++) {
				Neurons.TryGetValue(i, out Neuron outputNeuron);
				for (int j = 1; j < NumInputs; j++) {
					Neurons.TryGetValue(j, out Neuron inputNeuron);
					Tuple<int, int> fromToPair= new Tuple<int, int>(inputNeuron.ID, outputNeuron.ID);
					Connection conn;
					if (InnovationNumbers.ContainsKey(fromToPair))
					{
						lock (InnovationNumbers)
						{
							InnovationNumbers.TryGetValue(fromToPair, out int innoNumber);
							conn = new Connection(innoNumber, inputNeuron, rand.NextDouble() * 4 - 2);
						}
					}
					else
					{
						lock (InnovationNumbers)
						{
							conn = new Connection(InnovationNumbers.Count(), inputNeuron, rand.NextDouble() * 4 - 2);
							InnovationNumbers.Add(new Tuple<int, int>(inputNeuron.ID, outputNeuron.ID), InnovationNumbers.Count());
						}
					}
					Connections.Add(fromToPair);
					outputNeuron.AddConnection(conn.ID, conn);
				}
			}
		}

		/// <summary>
		///Evaluates all of the outputs of the neural network.
		/// </summary>
		/// <param name="inputs">Enumerable of all the input values</param>
		/// <returns>dictionary with names of outputs mapping to their values.</returns>
		public Dictionary<string, double> Evaluate(IEnumerable<double> inputs)
		{
			for (int i = 1; i < NumInputs; i++)
			{
				Neurons.TryGetValue(i, out Neuron inputNeuron);
				inputNeuron.Value = inputs.ElementAt(i-1);
			}
			Dictionary<string, double> outputs = new Dictionary<string, double>();
			for (int i = NumInputs; i < NumInputs + NumOutputs; i++)
			{
				Neurons.TryGetValue(i, out Neuron OutputNeuron);
				outputs.Add(outputNames[i - NumInputs], OutputNeuron.GetValue());
			}
			return outputs;
		}

		/// <summary>
		/// Mutates a random link whithin the neural network.
		/// </summary>
		public void MutateLink()
		{
			int from;
			int to;

			Stopwatch sw = new Stopwatch();
			sw.Start();

			do
			{
				rand = new Random(Guid.NewGuid().GetHashCode());
				do
				{
					rand = new Random(Guid.NewGuid().GetHashCode());
					from = rand.Next(Neurons.Count());
				} while (from > NumInputs - 1 && from < NumInputs + NumOutputs);

				do
				{
					rand = new Random(Guid.NewGuid().GetHashCode());
					to = rand.Next(NumInputs, Neurons.Count());
				} while (to == from || to < NumInputs || (to < from && to >= NumInputs+NumOutputs));
				if (sw.ElapsedMilliseconds > 10)
				{
					return;
				}
			} while (Connections.Contains(new Tuple<int, int>(from, to)));

			if (Neurons.TryGetValue(to, out Neuron toNeuron))
			{
				if (Neurons.TryGetValue(from, out Neuron fromNeuron))
				{
					Tuple<int, int> fromToPair = new Tuple<int, int>(from, to);
					Connection conn;
					if (InnovationNumbers.ContainsKey(fromToPair))
					{
						lock (InnovationNumbers)
						{
							InnovationNumbers.TryGetValue(fromToPair, out int innoNumber);
							conn = new Connection(innoNumber, fromNeuron, rand.NextDouble() * 4 - 2);
						}
					}
					else
					{
						lock (InnovationNumbers)
						{
							conn = new Connection(InnovationNumbers.Count(), fromNeuron, rand.NextDouble() * 4 - 2);
							InnovationNumbers.Add(new Tuple<int, int>(from, to), InnovationNumbers.Count());
						}
					}
					Connections.Add(fromToPair);
					toNeuron.AddConnection(conn.ID, conn);
				}
			}
		}

		/// <summary>
		/// mutates a random link in the neural network
		/// </summary>
		public void MutateNeuron()
		{
			while (true)
			{
				int neuronId = rand.Next(NumInputs, NumInputs + NumOutputs);
				if (Neurons.TryGetValue(neuronId, out Neuron farNeuron) && farNeuron.Connections.Count > 0)
				{
					Connection conn;
					int connectionId;
					do {
						connectionId = rand.Next(farNeuron.Connections.Count());
						conn = farNeuron.Connections.Values.ElementAt(connectionId);
					} while (conn.From.ID == 0);

					Neuron newNeuron = new Neuron(Neurons.Count);
					
					Tuple<int, int> nearToMidTuple = new Tuple<int, int>(conn.From.ID, newNeuron.ID);
					Connection nearToMid;
					lock (InnovationNumbers)
					{
						if (InnovationNumbers.TryGetValue(nearToMidTuple, out int innoNumber))
						{
							nearToMid = new Connection(innoNumber, conn.From, 1);
						}
						else
						{
							nearToMid = new Connection(InnovationNumbers.Count(), conn.From, 1);
							InnovationNumbers.Add(nearToMidTuple, InnovationNumbers.Count());
						}
					}
					Connection midToFar;
					Tuple<int, int> midToFarTuple = new Tuple<int, int>(newNeuron.ID, farNeuron.ID);
					lock (InnovationNumbers)
					{
						if (InnovationNumbers.TryGetValue(midToFarTuple, out int innoNumber2))
						{
							midToFar = new Connection(innoNumber2, newNeuron, conn.Weight);
						}
						else
						{
							midToFar = new Connection(InnovationNumbers.Count(), newNeuron, conn.Weight);
							InnovationNumbers.Add(midToFarTuple, InnovationNumbers.Count());
						}
					}

					Neurons.Add(newNeuron.ID, newNeuron);
					newNeuron.AddConnection(nearToMid.ID, nearToMid);
					Connection oldConnection = farNeuron.Connections.Values.ElementAt(connectionId);
					conn.EnableDisable();
					Connections.Add(nearToMidTuple);
					Connections.Add(midToFarTuple);

					farNeuron.AddConnection(midToFar.ID, midToFar);
					NumNeurons++;
					return;
				}
			}
		}

		/// <summary>
		/// Returns the count of all Neurons and all connections.
		/// </summary>
		/// <returns>The count of all neurons and all connections</returns>
		public int Size()
		{
			return Connections.Count() + Neurons.Count();
		}


		/// <summary>
		/// Returns a string representing the neural network.
		/// </summary>
		/// <returns>A string representing the neural network.</returns>
		public override string ToString()
		{
			{
				StringBuilder sb = new StringBuilder();
				foreach (Neuron neuron in Neurons.Values)
				{
					foreach (Connection conn in neuron.Connections.Values)
					{
						if (conn.Enabled)
						{
							sb.Append("(" + conn.From.ID + " -> " + neuron.ID + ", w=" + conn.Weight + ", inno=" + conn.ID + " enabled)\n");
						}
						else
						{
							sb.Append("(" + conn.From.ID + " -> " + neuron.ID + ", w=" + conn.Weight + ", inno=" + conn.ID + " disabled)\n");
						}
					}
				}
				return sb.ToString();
			}
		}

		/// <summary>
		/// Checks that all connections are represented in the Connections dictionary.
		/// </summary>
		/// <returns>True if all connections are represented in the Connections dictionary.</returns>
		public bool IsConsistent()
		{
			try
			{
				foreach (Tuple<int, int> connTuple in Connections)
				{
					bool match = false;
					foreach (Neuron neuron in Neurons.Values)
					{
						foreach (Connection conn in neuron.Connections.Values)
						{
							if (connTuple.Equals(new Tuple<int, int>(conn.From.ID, neuron.ID)))
							{
								match = true;
							}
						}
					}
					if (!match) {
						return false;
					}
				}
				return true;
			}
			catch (NullReferenceException) {
				return false;
			}
		}

		/// <summary>
		/// Returns a string in DOT language that can be used to visualize the neural network.
		/// </summary>
		/// <returns>A string in DOT language that can be used to visualize the neural network.</returns>
		public string GenerateDOT()
		{
			StringBuilder sb = new StringBuilder("digraph NeuralNetwork{\n");
			sb.Append("subgraph cluster_level1{\nlabel = \"Inputs\";\n");
			for (int i = 1; i < NumInputs; i++)
			{
				Neuron neur = Neurons.Values.ElementAt(i);
				sb.Append(neur.ID + ";\n");
			}
			sb.Append("}\nsubgraph cluster_level3{\nlabel = \"Outputs\";\n");
			for (int i = NumInputs; i < NumInputs + NumOutputs; i++)
			{
				Neuron neur = Neurons.Values.ElementAt(i);
				sb.Append(neur.ID + "[label=\"" + outputNames.ElementAt(i - NumInputs) + "\"];\n");
			}
			sb.Append("}\nsubgraph cluster_level2{\nlabel = \"Hidden Layers\";\nrankdir=LR;\n");
			for (int i = NumInputs + NumOutputs; i < Neurons.Count(); i++)
			{
				Neuron neur = Neurons.Values.ElementAt(i);
				sb.Append(neur.ID + ";\n");
			}
			sb.Append("}\n");
			foreach (Neuron neuron in Neurons.Values)
			{
				foreach (Connection conn in neuron.Connections.Values)
				{
					if (conn.Enabled)
					{
						sb.Append(conn.From.ID + " -> " + neuron.ID + "[label=\"" + Math.Round(conn.Weight, 3) + "\"];\n");
					}
					else
					{
						sb.Append(conn.From.ID + " -> " + neuron.ID + "[label=\"" + Math.Round(conn.Weight, 3) + "\", style=dotted];\n");
					}
				}
			}
			sb.Append("}");
			return sb.ToString();
		}

		/// <summary>
		/// Randomly mutates the neural network by changing weights, adding connections, and/or adding neurons.
		/// </summary>
		public void RandomMutation()
		{
			double connectionWeightsMutation = rand.NextDouble();
			double connectionMutation = rand.NextDouble();
			double NeuronMutation = rand.NextDouble();

			if (connectionWeightsMutation < connectionWeightsMutationChance)
			{
				for (int outputNeuronIdx = NumInputs; outputNeuronIdx < NumInputs + NumOutputs; outputNeuronIdx++)
				{
					Neurons.TryGetValue(outputNeuronIdx, out Neuron outputNeuron);
					MutateConnections(outputNeuron);
				}
			}

			if (connectionMutation < connectionMutationChance)
			{
				MutateLink();
			}

			if (NeuronMutation < NeuronMutationChance)
			{
				MutateNeuron();
			}
		}

		/// <summary>
		/// Mutates the connection weights all the neurons the provided neuron is dependent upon.
		/// </summary>
		/// <param name="neuron">The top most neuron</param>
		private void MutateConnections(Neuron neuron)
		{
			if (neuron.Connections.Count == 0)
			{
				return;
			}
			else
			{
				foreach (Connection conn in neuron.Connections.Values)
				{
					double perturbedChance = rand.NextDouble();
					double randomChance = rand.NextDouble();
					if (randomChance < .1)
					{
						conn.MutateWeightRandom();
						
					}
					if (perturbedChance < .9)
					{
						conn.MutateWeightShift();
					}

					MutateConnections(conn.From);
				}
			}
		}

		/// <summary>
		/// Performs crossover on two neural networks and returns the resulting network.
		/// </summary>
		/// <param name="NN1">first parent NN</param>
		/// <param name="NN2">second parent NN</param>
		/// <param name="InnovationNumbers">the innovation numbers used from a population</param>
		/// <returns>The resulting child NN</returns>
		public static NeuralNetwork Crossover(NeuralNetwork NN1, NeuralNetwork NN2, Dictionary<Tuple<int, int>, int> InnovationNumbers)
		{
			if (NN1.Fitness == double.MinValue || NN2.Fitness == double.MinValue)
			{
				throw new ArgumentException("One of the neural networks did not have a fitness assigned to it.");
			}
			Random rand = new Random(Guid.NewGuid().GetHashCode());
			NeuralNetwork childNetwork = new NeuralNetwork(NN1.NumInputs-1, NN1.outputNames, NN1.InnovationNumbers);

			NeuralNetwork fit;
			NeuralNetwork lessFit;
			if (NN1.Fitness > NN2.Fitness)
			{
				fit = NN1;
				lessFit = NN2;
			}
			else
			{
				fit = NN2;
				lessFit = NN1;
			}

			foreach (Tuple<int, int> conn in fit.Connections)
			{
				if (!childNetwork.Neurons.ContainsKey(conn.Item1))
				{
					childNetwork.Neurons.Add(conn.Item1, new Neuron(conn.Item1));
				}
				if (!childNetwork.Neurons.ContainsKey(conn.Item2))
				{
					childNetwork.Neurons.Add(conn.Item2, new Neuron(conn.Item2));
				}
				childNetwork.Neurons.TryGetValue(conn.Item2, out Neuron toNeuron);
				childNetwork.Neurons.TryGetValue(conn.Item1, out Neuron fromNeuron);
				lock (InnovationNumbers)
				{
					//Console.WriteLine("fit Consistency: " + fit.isConsistent());
					//Console.WriteLine("less fit Consistency: " + lessFit.isConsistent());
					InnovationNumbers.TryGetValue(conn, out int innoNumber);
					if (lessFit.Connections.Contains(conn))
					{
						fit.Neurons.TryGetValue(conn.Item2, out Neuron fitParentNeuron);
						fitParentNeuron.Connections.TryGetValue(innoNumber, out Connection fitSelectedCon);
						lessFit.Neurons.TryGetValue(conn.Item2, out Neuron lessFitParentNeuron);
						lessFitParentNeuron.Connections.TryGetValue(innoNumber, out Connection lessFitSelectedCon);
						if (rand.NextDouble() >= .5)
						{
							Connection newConn = new Connection(fitSelectedCon.ID, fromNeuron, fitSelectedCon.Weight);
							if (!fitSelectedCon.Enabled || !lessFitSelectedCon.Enabled)
							{
								double disableChance = rand.NextDouble();
								if (disableChance < .75) {
									newConn.EnableDisable();
								}
							}
							toNeuron.AddConnection(innoNumber, newConn);
						}
						else
						{
							Connection newConn = new Connection(lessFitSelectedCon.ID, fromNeuron, lessFitSelectedCon.Weight);
							if (!fitSelectedCon.Enabled || !lessFitSelectedCon.Enabled)
							{
								double disableChance = rand.NextDouble();
								if (disableChance < .75)
								{
									newConn.EnableDisable();
								}
							}
							toNeuron.AddConnection(innoNumber, newConn);
						}
					}
					else
					{
						fit.Neurons.TryGetValue(conn.Item2, out Neuron parentNeuron);
						parentNeuron.Connections.TryGetValue(innoNumber, out Connection selectedCon);
						toNeuron.AddConnection(innoNumber, new Connection(selectedCon.ID, fromNeuron, selectedCon.Weight));
					}
				}
			}
			childNetwork.NumNeurons = childNetwork.Neurons.Count;
			childNetwork.Connections = new List<Tuple<int, int>>(fit.Connections);
			return childNetwork;
		}

		/// <summary>
		/// Returns the evolutionary distance between the provided NN's
		/// </summary>
		/// <param name="NN1">the first NN</param>
		/// <param name="NN2">the second NN</param>
		/// <param name="InnovationNumbers">the innovation numbers used from the population.</param>
		/// <returns></returns>
		public static double EvolutionaryDistance(NeuralNetwork NN1, NeuralNetwork NN2, Dictionary<Tuple<int, int>, int> InnovationNumbers)
		{
			int N = Math.Max(NN1.Size(), NN2.Size());

			double avgWeightDiff = 0;
			int numMatchingGenes = 0;

			List<int> NN1Disjoints = new List<int>();
			List<int> NN2Disjoints = new List<int>();

			foreach (Tuple<int, int> conn in NN1.Connections)
			{
				lock (InnovationNumbers)
				{
					if (NN2.Connections.Contains(conn))
					{
						numMatchingGenes++;

						InnovationNumbers.TryGetValue(conn, out int innoNumber);
						NN1.Neurons.TryGetValue(conn.Item2, out Neuron NN1Neuron);
						NN2.Neurons.TryGetValue(conn.Item2, out Neuron NN2Neuron);

						NN1Neuron.Connections.TryGetValue(innoNumber, out Connection conn1);
						NN2Neuron.Connections.TryGetValue(innoNumber, out Connection conn2);

						avgWeightDiff += Math.Abs(conn1.Weight - conn2.Weight);
					}
					else
					{
						InnovationNumbers.TryGetValue(conn, out int innoNumber);
						NN1Disjoints.Add(innoNumber);
					}
				}
			}
			avgWeightDiff = avgWeightDiff / numMatchingGenes;

			foreach (Tuple<int, int> conn in NN2.Connections)
			{

				if (!NN1.Connections.Contains(conn))
				{
					lock (InnovationNumbers)
					{
						InnovationNumbers.TryGetValue(conn, out int innoNumber);
						NN2Disjoints.Add(innoNumber);
					}
				}
			}

			int numExcessGenes = Math.Abs(NN1Disjoints.Count() - NN2Disjoints.Count());
			int numDisjointGenes = NN1Disjoints.Count() + NN2Disjoints.Count() - numExcessGenes;

			return (excessCoef * numExcessGenes) / N + (disjointCoef * numDisjointGenes) / N + avgWeightDiffCoef * avgWeightDiff;
		}
	}

	/// <summary>
	/// Represents a single Neuron in a Neural Network.
	/// </summary>
	public class Neuron
	{
		public int ID { get; private set; }
		public double Value;
		public Dictionary<int, Connection> Connections { get; private set; }

		/// <summary>
		/// Initializes a neuron with an id and a value
		/// </summary>
		/// <param name="_id">The id the neuron should be assigned</param>
		/// <param name="_value">The value of the neuron</param>
		public Neuron(int _id, double _value) : this(_id)
		{
			Value = _value;
		}

		/// <summary>
		/// Initializes a neuron with an id
		/// </summary>
		/// <param name="_id">The id the neuron should be assigned</param>
		public Neuron(int _id)
		{
			ID = _id;
			Connections = new Dictionary<int, Connection>();
		}

		/// <summary>
		/// adds a connection to the neurons list of connections.
		/// </summary>
		/// <param name="id">the id of the connection</param>
		/// <param name="connection">the connection to be added</param>
		/// <returns></returns>
		public bool AddConnection(int id, Connection connection)
		{
			if (!Connections.ContainsKey(id))
			{
				Connections.Add(id, connection);
				return true;
			}
			else
			{
				return false;
			}
		}

		/// <summary>
		/// Returns the value of the neuron, the weighted sums and biases of all previous neurons.
		/// </summary>
		/// <returns>The value of the neuron</returns>
		public double GetValue()
		{
			if (Connections.Count == 0)
			{
				return this.Value;
			}
			else
			{
				double value = 0;
				foreach (Connection conn in Connections.Values)
				{
					if (conn.Enabled) {
						value += conn.From.GetValue() * conn.Weight;
					}
				}
				return Sigmoid(value);
			}
		}

		/// <summary>
		/// Returns the sigmoid output for the given input.
		/// </summary>
		/// <param name="rawValue"></param>
		/// <returns>The sigmoid output for the given input (between 0 and 1)</returns>
		private double Sigmoid(double rawValue)
		{
			return 1 / (1 + Math.Exp(-4.9 * rawValue));
		}
	}

	/// <summary>
	/// represents a coneection between two neurons.
	/// </summary>
	public class Connection
	{
		public bool Enabled { get; private set; }
		public int ID { get; private set; }
		public Neuron From { get; set; }
		public double Weight { get; private set; }

		Random rand = new Random(Guid.NewGuid().GetHashCode());

		/// <summary>
		/// Constructs a connection
		/// </summary>
		/// <param name="_id">ID of the connection.</param>
		/// <param name="_from">The neuron the connection is from.</param>
		/// <param name="_weight">The weight of the connection.</param>
		public Connection(int _id, Neuron _from, double _weight)
		{
			ID = _id;
			From = _from;
			Weight = _weight;
			Enabled = true;
		}

		/// <summary>
		/// Randomly assigns a new weight to the connection.
		/// </summary>
		public void MutateWeightRandom()
		{
			Weight = rand.NextDouble() * 4 - 2;
		}

		/// <summary>
		/// randomly shifts the weight of the connection
		/// </summary>
		public void MutateWeightShift()
		{
			double shift = rand.NextDouble()*2 - 1;
			Weight += shift;
		}

		/// <summary>
		/// Enables the connection if it is disabled and vice versa.
		/// </summary>
		public void EnableDisable()
		{
			Enabled = !Enabled;
		}
	}
}

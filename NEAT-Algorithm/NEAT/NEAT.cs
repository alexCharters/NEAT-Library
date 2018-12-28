using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NEAT
{
	public class Population
	{
		readonly Dictionary<Tuple<int, int>, int> innovationNumbers;
		public delegate double analysisFunction(NeuralNetwork NN);
		private readonly analysisFunction run;
		public List<NeuralNetwork> Networks { get; private set; }
		public double SpeciatingThreshold = 1.2;
		public double killRate = .5;
		public List<Species> speciesList;
		public double AvgPopFitness { get; private set; }
		public int Size { get; private set; }

		public Population(IEnumerable<double> inputs, IEnumerable<string> outputNames, int size, analysisFunction _run)
		{
			run = _run;
			innovationNumbers = new Dictionary<Tuple<int, int>, int>();
			Networks = new List<NeuralNetwork>();
			speciesList = new List<Species>();
			Size = size;
			for (int i = 0; i < size; i++)
			{
				NeuralNetwork NN = new NeuralNetwork(inputs, outputNames, innovationNumbers);
				NN.InitializeRandom();
				Networks.Add(NN);
			}
		}

		public Population(int numInputs, IEnumerable<string> outputNames, int size, analysisFunction _run)
		{
			run = _run;
			innovationNumbers = new Dictionary<Tuple<int, int>, int>();
			Networks = new List<NeuralNetwork>();
			speciesList = new List<Species>();
			Size = size;
			for (int i = 0; i < size; i++)
			{
				NeuralNetwork NN = new NeuralNetwork(numInputs, outputNames, innovationNumbers);
				NN.InitializeRandom();
				Networks.Add(NN);
			}
		}

		public void Run()
		{
			List<double> fitnesses = new List<double>();
			int count = 0;
			double totalfitness = 0;
			foreach (NeuralNetwork NN in Networks)
			{
				Interlocked.Increment(ref count);
				double fitness = run(NN);
				totalfitness += fitness;
				fitnesses.Add(fitness);
				Console.WriteLine(count + " -> " + fitness);
			}

			AvgPopFitness = totalfitness / Networks.Count;

			for (int i = 0; i < Networks.Count(); i++)
			{
				Networks.ElementAt(i).Fitness = fitnesses.ElementAt(i);
			}
		}

		public void Select()
		{
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
			speciesList.RemoveAll(x => x.networks.Count == 1);
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

			int numSelectionBreed = (int)(.75 * Size);

			speciesList.Sort((species1, species2) => species2.adjustedFitnessSum.CompareTo(species1.adjustedFitnessSum));

			double sharedFitnessTotal = 0;
			for (int i = 0; i < speciesList.Count / 3; i++)
			{
				Species species = speciesList.ElementAt(i);
				sharedFitnessTotal += species.adjustedFitnessSum;
			}

			List<NeuralNetwork> childrenNetworks = new List<NeuralNetwork>();
			Random rand = new Random();
			for (int i = 0; i < speciesList.Count / 3; i++)
			{
				Species species = speciesList.ElementAt(i);
				for (int j = 0; j < numSelectionBreed * (species.adjustedFitnessSum / sharedFitnessTotal); j++)
				{
					NeuralNetwork NN1 = species.networks.ElementAt(rand.Next(species.networks.Count));
					NeuralNetwork NN2 = species.networks.ElementAt(rand.Next(species.networks.Count));
					NeuralNetwork Child = NeuralNetwork.Crossover(NN1, NN2, innovationNumbers);
					Child.RandomMutation();
					childrenNetworks.Add(Child);
				}
			}

			for (int i = 0; i < Networks.Count - childrenNetworks.Count; i++)
			{
				Species randSpecies = speciesList[rand.Next(speciesList.Count)];

				NeuralNetwork randNN1 = randSpecies.networks[rand.Next(randSpecies.networks.Count)];
				NeuralNetwork randNN2 = randSpecies.networks[rand.Next(randSpecies.networks.Count)];
				NeuralNetwork child = NeuralNetwork.Crossover(randNN1, randNN2, innovationNumbers);
				child.RandomMutation();

				childrenNetworks.Add(child);
			}
		}
	}

	public class Species
	{
		public double adjustedFitnessSum;
		public List<NeuralNetwork> networks;

		public Species(NeuralNetwork NN)
		{
			networks = new List<NeuralNetwork>() { NN };
		}
	}

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

		Random rand = new Random(Guid.NewGuid().GetHashCode());

		public NeuralNetwork(IEnumerable<double> inputs, IEnumerable<string> outputNames, Dictionary<Tuple<int, int>, int> _innovationNumbers)
		{
			this.outputNames = (List<string>)outputNames;
			Neurons = new Dictionary<int, Neuron>();
			Connections = new List<Tuple<int, int>>();
			InnovationNumbers = _innovationNumbers;
			NumInputs = inputs.Count();
			NumOutputs = outputNames.Count();
			NumNeurons = NumInputs + NumOutputs;
			Fitness = double.MinValue;
			for (int id = 0; id < NumInputs; id++)
			{
				Neurons.Add(id, new Neuron(id, inputs.ElementAt(id)));
			}

			for (int id = 0; id < NumOutputs; id++)
			{
				Neurons.Add(id + NumInputs, new Neuron(id + NumInputs));
			}
		}

		public NeuralNetwork(int numInputs, IEnumerable<string> outputNames, Dictionary<Tuple<int, int>, int> _innovationNumbers)
		{
			this.outputNames = (List<string>)outputNames;
			Neurons = new Dictionary<int, Neuron>();
			Connections = new List<Tuple<int, int>>();
			InnovationNumbers = _innovationNumbers;
			NumInputs = numInputs;
			NumOutputs = outputNames.Count();
			NumNeurons = NumInputs + NumOutputs;
			Fitness = double.MinValue;
			for (int id = 0; id < NumInputs; id++)
			{
				Neurons.Add(id, new Neuron(id));
			}

			for (int id = 0; id < NumOutputs; id++)
			{
				Neurons.Add(id + NumInputs, new Neuron(id + NumInputs));
			}
		}

		public void InitializeRandom()
		{
			int round1LinksNum = Math.Max((int)(rand.NextDouble() * NumInputs - 2), 1);
			//int round1NeuronsNum = (int)(rand.NextDouble() * NumOutputs * 2) + 1;
			//int round2LinksNum = (int)(rand.NextDouble() * NumInputs);

			for (int i = 0; i < round1LinksNum; i++)
			{
				MutateLink();
			}
			//for (int i = 0; i < round1NeuronsNum; i++)
			//{
			//	MutateNeuron();
			//}
			//for (int i = 0; i < round2LinksNum; i++)
			//{
			//	MutateLink();
			//}
		}

		public Dictionary<string, double> Evaluate(IEnumerable<double> inputs)
		{
			for (int i = 0; i < NumInputs; i++)
			{
				Neurons.TryGetValue(i, out Neuron inputNeuron);
				inputNeuron.Value = inputs.ElementAt(i);
			}
			Dictionary<string, double> outputs = new Dictionary<string, double>();
			for (int i = NumInputs; i < NumInputs + NumOutputs; i++)
			{
				Neurons.TryGetValue(i, out Neuron OutputNeuron);
				outputs.Add(outputNames[i - NumInputs], OutputNeuron.GetValue());
			}
			return outputs;
		}

		public void MutateLink()
		{
			int from;
			int to;

			do
			{
				do
				{
					from = rand.Next(Neurons.Count());
				} while (from > NumInputs - 1 && from < NumInputs + NumOutputs);

				do
				{
					to = rand.Next(Neurons.Count());
				} while (to == from || to < NumInputs || (to > NumInputs + NumOutputs && to < from));
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

		public void MutateNeuron()
		{
			while (true)
			{
				int neuronId = rand.Next(NumInputs, NumInputs + NumOutputs);
				if (Neurons.TryGetValue(neuronId, out Neuron farNeuron) && farNeuron.Connections.Count > 0)
				{
					int connectionId = rand.Next(farNeuron.Connections.Count());
					Neuron newNeuron = new Neuron(NumNeurons);
					Connection conn = farNeuron.Connections.Values.ElementAt(connectionId);
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
					Connection midtoFar;
					Tuple<int, int> midToFarTuple = new Tuple<int, int>(newNeuron.ID, farNeuron.ID);
					lock (InnovationNumbers)
					{
						if (InnovationNumbers.TryGetValue(midToFarTuple, out int innoNumber2))
						{
							midtoFar = new Connection(innoNumber2, newNeuron, conn.Weight);
						}
						else
						{
							midtoFar = new Connection(InnovationNumbers.Count(), newNeuron, conn.Weight);
							InnovationNumbers.Add(midToFarTuple, InnovationNumbers.Count());
						}
					}

					Neurons.Add(newNeuron.ID, newNeuron);
					newNeuron.AddConnection(nearToMid.ID, nearToMid);
					Connection oldConnection = farNeuron.Connections.Values.ElementAt(connectionId);
					conn.EnableDisable();
					Connections.Add(nearToMidTuple);
					Connections.Add(midToFarTuple);

					farNeuron.AddConnection(midtoFar.ID, midtoFar);
					NumNeurons++;
					return;
				}
			}
		}

		public int Size()
		{
			return Connections.Count() + Neurons.Count();
		}

		public override string ToString()
		{
			{
				StringBuilder sb = new StringBuilder();
				foreach (Neuron neuron in Neurons.Values)
				{
					foreach (Connection conn in neuron.Connections.Values)
					{
						sb.Append("(" + conn.From.ID + " -> " + neuron.ID + ", w=" + conn.Weight + ", inno=" + conn.ID + ")\n");
					}
				}
				return sb.ToString();
			}
		}

		public string GenerateDOT()
		{
			StringBuilder sb = new StringBuilder("digraph NeuralNetwork{\n");
			sb.Append("subgraph cluster_level1{\nlabel = \"Inputs\";\n");
			for (int i = 0; i < NumInputs; i++)
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

		public void RandomMutation()
		{
			double connectionWeightsMutationChance = rand.NextDouble();
			double connectionMutationChance = rand.NextDouble();
			double NeuronMutationChance = rand.NextDouble();

			if (connectionWeightsMutationChance < .8)
			{
				for (int outputNeuronIdx = NumInputs; outputNeuronIdx < NumInputs + NumOutputs; outputNeuronIdx++)
				{
					Neurons.TryGetValue(outputNeuronIdx, out Neuron outputNeuron);
					MutateConnections(outputNeuron);
				}
			}

			if (connectionMutationChance < .05)
			{
				MutateLink();
			}

			if (NeuronMutationChance < .03)
			{
				MutateNeuron();
			}
		}

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
					double perturbedOrRandom = rand.NextDouble();
					if (perturbedOrRandom < .9)
					{
						conn.MutateWeightShift();
					}
					else
					{
						conn.MutateWeightRandom();
					}

					MutateConnections(conn.From);
				}
			}
		}

		public static NeuralNetwork Crossover(NeuralNetwork NN1, NeuralNetwork NN2, Dictionary<Tuple<int, int>, int> InnovationNumbers)
		{
			if (NN1.Fitness == double.MinValue || NN2.Fitness == double.MinValue)
			{
				throw new ArgumentException("One of the neural networks did not have a fitness assigned to it.");
			}
			Random rand = new Random(Guid.NewGuid().GetHashCode());
			NeuralNetwork childNetwork = new NeuralNetwork(NN1.NumInputs, NN1.outputNames, NN1.InnovationNumbers);

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
				lock (InnovationNumbers)
				{
					InnovationNumbers.TryGetValue(conn, out int innoNumber);
					if (lessFit.Connections.Contains(conn))
					{
						if (rand.NextDouble() >= .5)
						{
							fit.Neurons.TryGetValue(conn.Item2, out Neuron parentNeuron);
							parentNeuron.Connections.TryGetValue(innoNumber, out Connection selectedCon);
							toNeuron.AddConnection(innoNumber, selectedCon);
						}
						else
						{
							lessFit.Neurons.TryGetValue(conn.Item2, out Neuron parentNeuron);
							parentNeuron.Connections.TryGetValue(innoNumber, out Connection selectedCon);
							toNeuron.AddConnection(innoNumber, selectedCon);
						}
					}
					else
					{
						fit.Neurons.TryGetValue(conn.Item2, out Neuron parentNeuron);
						parentNeuron.Connections.TryGetValue(innoNumber, out Connection selectedCon);
						toNeuron.Connections.Add(innoNumber, selectedCon);
					}
				}
			}
			return childNetwork;
		}

		public static double EvolutionaryDistance(NeuralNetwork NN1, NeuralNetwork NN2, Dictionary<Tuple<int, int>, int> InnovationNumbers)
		{
			int N = Math.Max(NN1.Size(), NN2.Size());
			double disjointCoef = 1.0;
			double excessCoef = 1.0;
			double avgWeightDiffCoef = 0.4;

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

	public class Neuron
	{
		public int ID { get; private set; }
		public double Value;
		public Dictionary<int, Connection> Connections { get; private set; }

		public Neuron(int _id, double _value) : this(_id)
		{
			Value = _value;
		}

		public Neuron(int _id)
		{
			ID = _id;
			Connections = new Dictionary<int, Connection>();
		}

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
					value += conn.From.GetValue() * conn.Weight;
				}
				return Sigmoid(value);
			}
		}

		private double Sigmoid(double rawValue)
		{
			return 1 / (1 + Math.Exp(-rawValue));
		}
	}

	public class Connection
	{
		public bool Enabled { get; private set; }
		public int ID { get; private set; }
		public Neuron From { get; set; }
		public double Weight { get; private set; }

		Random rand = new Random(Guid.NewGuid().GetHashCode());

		public Connection(int _id, Neuron _from, double _weight)
		{
			ID = _id;
			From = _from;
			Weight = _weight;
			Enabled = true;
		}

		public void MutateWeightRandom()
		{
			Weight = rand.NextDouble() * 4 - 2;
		}

		public void MutateWeightShift()
		{
			double shift = rand.NextDouble() - 0.5;
			Weight += shift;
		}

		public void EnableDisable()
		{
			Enabled = !Enabled;
		}
	}
}

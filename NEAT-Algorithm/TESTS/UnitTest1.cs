using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NEAT;

namespace NEAT_Tests
{
	[TestClass]
	public class UnitTest1
	{
		[TestMethod]
		public void TupleHashsetContainsTest()
		{
			HashSet<Tuple<int, int>> hash = new HashSet<Tuple<int, int>>
			{
				new Tuple<int, int>(1, 7)
			};
			Assert.IsTrue(hash.Contains(new Tuple<int, int>(1, 7)));
		}

		[TestMethod]
		public void TupleHashcodeTest()
		{
			Tuple<int, int> t1 = new Tuple<int, int>(2, 9);
			Tuple<int, int> t2 = new Tuple<int, int>(2, 9);

			Assert.AreEqual(t1.GetHashCode(), t2.GetHashCode());
		}

		[TestMethod]
		public void GetNeuronValueTest()
		{
			Neuron childNeuron1 = new Neuron(0, 3);
			Neuron childNeuron2 = new Neuron(1, 5);
			Neuron parentNeuron = new Neuron(2);

			Connection conn1 = new Connection(0, childNeuron1, -5);
			Connection conn2 = new Connection(1, childNeuron2, 3);

			parentNeuron.AddConnection(0, conn1);
			parentNeuron.AddConnection(1, conn2);

			Assert.IsTrue(Math.Abs(parentNeuron.GetValue() - 0.5) < .0000000001);
		}

		public List<double> GetInputs()
		{
			return new List<double>
			{
				12,
				1,
				2,
				7,
				2,
				12,
				12,
				12,
				12,
				12
			};
		}

		public List<string> GetOutputs()
		{
			return new List<string>
			{
				"W",
				"A",
				"S",
				"D"
			};
		}
	}
}

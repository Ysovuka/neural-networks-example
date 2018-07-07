using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.IO;

namespace console
{
    class Program
    {
        private int _hiddenDims = 2;        // Number of hidden neurons.
        private int _inputDims = 2;        // Number of input neurons.
        private int _iteration;            // Current training iteration.
        private int _restartAfter = 2000;   // Restart training if iterations exceed this.
        private Layer _hidden;              // Collection of hidden neurons.
        private Layer _inputs;              // Collection of input neurons.
        private List<Pattern> _patterns;    // Collection of training patterns.
        private Neuron _output;            // Output neuron.
        private Random _rnd = new Random(); // Global random number generator.
        [STAThread]

        static void Main()
        {
            new Program();
        }

        public Program()
        {
            LoadPatterns();
            Initialise();
            Train();
            Test();
        }
        private void Train()
        {
            double error;
            do
            {
                error = 0;
                foreach (Pattern pattern in _patterns)
                {
                    double delta = pattern.Output - Activate(pattern);
                    AdjustWeights(delta);
                    error += Math.Pow(delta, 2);
                }
                Console.WriteLine("Iteration {0} Error {1:0.000}", _iteration, error);
                _iteration++;
                if (_iteration > _restartAfter) Initialise();
            } while (error > 0.1);
        }
        private void Test()
        {
            Console.WriteLine("Begin network testing\nPress Ctrl C to exit");
            while (1 == 1)
            {
                try
                {
                    Console.Write("Input x, y: ");
                    string values = Console.ReadLine() + ",0";
                    Console.WriteLine("{0:0}", Activate(new Pattern(values, _inputDims)));
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
        }
        private double Activate(Pattern pattern)
        {
            for (int i = 0; i < pattern.Inputs.Length; i++)
            {
                _inputs[i].Output = pattern.Inputs[i];
            }
            foreach (Neuron neuron in _hidden)
            {
                neuron.Activate();
            }
            _output.Activate();
            return _output.Output;
        }
        private void AdjustWeights(double delta)
        {
            _output.AdjustWeights(delta);
            foreach (Neuron neuron in _hidden)
            {
                neuron.AdjustWeights(_output.ErrorFeedback(neuron));
            }
        }
        private void Initialise()
        {
            _inputs = new Layer(_inputDims);
            _hidden = new Layer(_hiddenDims, _inputs, _rnd);
            _output = new Neuron(_hidden, _rnd);
            _iteration = 0;
            Console.WriteLine("Network Initialised");
        }
        private void LoadPatterns()
        {
            _patterns = new List<Pattern>();
            StreamReader file = File.OpenText("Patterns.csv");
            while (!file.EndOfStream)
            {
                string line = file.ReadLine();
                _patterns.Add(new Pattern(line, _inputDims));
            }
            file.Close();
        }

        //static void Main(string[] args)
        //{
        //    // Load sample input patterns.
        //    double[,] inputs = new double[,] {
        //        { 0.72, 0.82 }, { 0.91, -0.69 }, { 0.46, 0.80 },
        //        { 0.03, 0.93 }, { 0.12, 0.25 }, { 0.96, 0.47 },
        //        { 0.79, -0.75 }, { 0.46, 0.98 }, { 0.66, 0.24 },
        //        { 0.72, -0.15 }, { 0.35, 0.01 }, { -0.16, 0.84 },
        //        { -0.04, 0.68 }, { -0.11, 0.10 }, { 0.31, -0.96 },
        //        { 0.00, -0.26 }, { -0.43, -0.65 }, { 0.57, -0.97 },
        //        { -0.47, -0.03 }, { -0.72, -0.64 }, { -0.57, 0.15 },
        //        { -0.25, -0.43 }, { 0.47, -0.88 }, { -0.12, -0.90 },
        //        { -0.58, 0.62 }, { -0.48, 0.05 }, { -0.79, -0.92 },
        //        { -0.42, -0.09 }, { -0.76, 0.65 }, { -0.77, -0.76 } };
        //    // Load sample output patterns.
        //    int[] outputs = new int[] {
        //        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        //        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        //    int patternCount = inputs.GetUpperBound(0) + 1;
        //    // Randomise weights.
        //    //Random r = new Random();
        //    //double[] weights = { r.NextDouble(), r.NextDouble() };
        //    // Set learning rate.
        //    //double learningRate = 0.1;
        //    SingleLayerPerception neuron = new SingleLayerPerception();

        //    int iteration = 0;
        //    double globalError;
        //    do
        //    {
        //        globalError = 0;
        //        for (int p = 0; p < patternCount; p++)
        //        {
        //            neuron.ClearInputs();
        //            neuron.AddInput(inputs[p, 0]);
        //            neuron.AddInput(inputs[p, 1]);

        //            neuron.Initialize();

        //            // Calculate output.
        //            double output = neuron.Learn();
        //            // Calculate error.
        //            double localError = outputs[p] - output;
        //            if (localError != 0)
        //            {
        //                // Update weights.
        //                neuron.Teach(localError);
        //            }

        //            Console.WriteLine("Local Error: {0}", localError);
        //            // Convert error to absolute value.
        //            globalError += Math.Abs(localError);
        //        }
        //        Console.WriteLine("Iteration {0} Error {1}", iteration, globalError);
        //        iteration++;
        //    } while (globalError != 0);
        //    // Display network generalisation.
        //    Console.WriteLine();
        //    Console.WriteLine("X, Y, Output");
        //    for (double x = -1; x <= 1; x += .5)
        //    {
        //        for (double y = -1; y <= 1; y += .5)
        //        {
        //            neuron.ClearInputs();
        //            neuron.AddInput(x);
        //            neuron.AddInput(y);

        //            // Calculate output.
        //            double output = neuron.Learn();
        //            Console.WriteLine("{0}, {1}, {2}", x, y, (output == 1) ? "Blue" : "Red");
        //        }
        //    }

        //    Console.ReadKey();
        //}
        //private static int Output(double[] weights, double x, double y)
        //{
        //    double sum = x * weights[0] + y * weights[1];
        //    return (sum >= 0) ? 1 : -1;
        //}
    }
}

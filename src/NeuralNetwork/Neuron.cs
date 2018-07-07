using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Layer : List<Neuron>
    {
        public Layer(int size)
        {
            for (int i = 0; i < size; i++)
                Add(new Neuron());
        }
        public Layer(int size, Layer layer, Random rnd)
        {
            for (int i = 0; i < size; i++)
                Add(new Neuron(layer, rnd));
        }
    }

    public class Neuron
    {
        private double _bias;                       // Bias value.
        private double _error;                      // Sum of error.
        private double _input;                      // Sum of inputs.
        private double _lambda = 6;                // Steepness of sigmoid curve.
        private double _learnRate = 0.5;            // Learning rate.
        private double _output = double.MinValue;   // Preset value of neuron.
        private List<Weight> _weights;              // Collection of weights to inputs.
        public Neuron() { }
        public Neuron(Layer inputs, Random rnd)
        {
            _weights = new List<Weight>();
            foreach (Neuron input in inputs)
            {
                Weight w = new Weight
                {
                    Input = input,
                    Value = rnd.NextDouble() * 2 - 1
                };
                _weights.Add(w);
            }
        }
        public void Activate()
        {
            _input = 0;
            foreach (Weight w in _weights)
            {
                _input += w.Value * w.Input.Output;
            }
        }
        public double ErrorFeedback(Neuron input)
        {
            Weight w = _weights.Find(delegate (Weight t) { return t.Input == input; });
            return _error * Derivative * w.Value;
        }
        public void AdjustWeights(double value)
        {
            _error = value;
            for (int i = 0; i < _weights.Count; i++)
            {
                _weights[i].Value += _error * Derivative * _learnRate * _weights[i].Input.Output;
            }
            _bias += _error * Derivative * _learnRate;
        }
        private double Derivative
        {
            get
            {
                double activation = Output;
                return activation * (1 - activation);
            }
        }
        public double Output
        {
            get
            {
                if (_output != double.MinValue)
                {
                    return _output;
                }
                return 1 / (1 + Math.Exp(-_lambda * (_input + _bias)));
            }
            set
            {
                _output = value;
            }
        }
    }


    public class Pattern
    {
        private double[] _inputs;
        private double _output;
        public Pattern(string value, int inputSize)
        {
            string[] line = value.Split(',');
            if (line.Length - 1 != inputSize)
                throw new Exception("Input does not match network configuration");
            _inputs = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                _inputs[i] = double.Parse(line[i]);
            }
            _output = double.Parse(line[inputSize]);
        }
        public double[] Inputs
        {
            get { return _inputs; }
        }
        public double Output
        {
            get { return _output; }
        }
    }

    public class Weight
    {
        public Neuron Input;
        public double Value;
    }

}

//public class Network
//{
//    private int _hiddenDims = 2;        // Number of hidden neurons.
//    private int _inputDims = 2;        // Number of input neurons.
//    private int _iteration;            // Current training iteration.
//    private int _restartAfter = 2000;   // Restart training if iterations exceed this.
//    private Layer _hidden;              // Collection of hidden neurons.
//    private Layer _inputs;              // Collection of input neurons.
//    private List<Pattern> _patterns;    // Collection of training patterns.
//    private Neuron _output;            // Output neuron.
//    private Random _rnd = new Random(); // Global random number generator.
//    [STAThread]
//    static void Main()
//    {
//        new Network();
//    }
//    public Network()
//    {
//        LoadPatterns();
//        Initialise();
//        Train();
//        Test();
//    }
//    private void Train()
//    {
//        double error;
//        do
//        {
//            error = 0;
//            foreach (Pattern pattern in _patterns)
//            {
//                double delta = pattern.Output – Activate(pattern);
//                AdjustWeights(delta);
//                error += Math.Pow(delta, 2);
//            }
//            Console.WriteLine(“Iteration { 0}
//            tError { 1:0.000}”, _iteration, error);
//            _iteration++;
//            if (_iteration > _restartAfter) Initialise();
//        } while (error > 0.1);
//    }
//    private void Test()
//    {
//        Console.WriteLine(“nBegin network testingnPress Ctrl C to exitn”);
//        while (1 == 1)
//        {
//            try
//            {
//                Console.Write(“Input x, y: “);
//                string values = Console.ReadLine() + “,0”;
//                Console.WriteLine(“{ 0:0}
//                n”, Activate(new Pattern(values, _inputDims)));
//            }
//            catch (Exception e)
//            {
//                Console.WriteLine(e.Message);
//            }
//        }
//    }
//    private double Activate(Pattern pattern)
//    {
//        for (int i = 0; i < pattern.Inputs.Length; i++)
//        {
//            _inputs[i].Output = pattern.Inputs[i];
//        }
//        foreach (Neuron neuron in _hidden)
//        {
//            neuron.Activate();
//        }
//        _output.Activate();
//        return _output.Output;
//    }
//    private void AdjustWeights(double delta)
//    {
//        _output.AdjustWeights(delta);
//        foreach (Neuron neuron in _hidden)
//        {
//            neuron.AdjustWeights(_output.ErrorFeedback(neuron));
//        }
//    }
//    private void Initialise()
//    {
//        _inputs = new Layer(_inputDims);
//        _hidden = new Layer(_hiddenDims, _inputs, _rnd);
//        _output = new Neuron(_hidden, _rnd);
//        _iteration = 0;
//        Console.WriteLine(“Network Initialised”);
//    }
//    private void LoadPatterns()
//    {
//        _patterns = new List<Pattern>();
//        StreamReader file = File.OpenText(“Patterns.csv”);
//        while (!file.EndOfStream)
//        {
//            string line = file.ReadLine();
//            _patterns.Add(new Pattern(line, _inputDims));
//        }
//        file.Close();
//    }
//}


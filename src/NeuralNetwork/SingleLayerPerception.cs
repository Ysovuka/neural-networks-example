using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class SingleLayerPerception
    {
        private bool _initialized = false;
        private const double LEARNING_RATE = 0.1;
        private Random _random = new Random();

        private IList<double> _inputs = new List<double>();
        private IList<double> _weights = new List<double>();

        public SingleLayerPerception() { }

        public void AddInput(double input) => _inputs.Add(input);
        public void Initialize()
        {
            for(int i = 0; i < _inputs.Count; i++)
            {
                _weights.Add(_random.NextDouble());
            }

            _initialized = true;
        }

        public void ClearInputs() { _inputs.Clear(); }
       
        public double Learn()
        {
            if (!_initialized) { Initialize(); }

            double output = 0.0;
            for(int i = 0; i < _inputs.Count; i++)
            {
                output += TrainInputWithWeight(_inputs[i], _weights[i]);
            }

            return Sigmoid(output);
        }

        public void Teach(double error)
        {
            //for (int w = 0; w < _weights.Count; w++)
            for (int i = 0; i < _inputs.Count; i++)
            {
                _weights[i] += LEARNING_RATE * error * _inputs[i];
            }
        }

        private double TrainInputWithWeight(double input, double weight)
        {
            return input * weight;
        }

        private double Sigmoid(double x)
        {
            return 2 / (1 + Math.Exp(-6 * x)) - 1;
        }

        //private double Sigmoid(double x)
        //{
        //    return 1 / (1 + Math.Exp(-x));
        //}

        private double Derivative(double x)
        {
            double s = Sigmoid(x);
            return 1 - (Math.Pow(s, 2));
        }

        //public double Derivative(double x)
        //{
        //    double s = Sigmoid(x);
        //    return s * (1 - s);
        //}
    }
}

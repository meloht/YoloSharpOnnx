using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class YoloSharp : IDisposable
    {
        private IExecutionProvider _executionProvider;
        public InterpolationFlags ResizeAlgorithm { get; set; } = InterpolationFlags.Linear;

        public YoloSharp(IExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}

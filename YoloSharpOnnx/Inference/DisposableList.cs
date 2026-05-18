using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public class DisposableList<T> : List<T>, IDisposableReadOnlyCollection<T> where T : IDisposable
    {
        public DisposableList()
        { }

        public DisposableList(IEnumerable<T> enumerable) : base(enumerable)
        { }

        public DisposableList(int count)
            : base(count)
        { }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // Dispose in the reverse order.
                    // Objects should typically be destroyed/disposed
                    // in the reverse order of its creation
                    // especially if the objects created later refer to the
                    // objects created earlier. For homogeneous collections of objects
                    // it would not matter.
                    for (int i = this.Count - 1; i >= 0; --i)
                    {
                        this[i]?.Dispose();
                    }
                    this.Clear();
                }

                disposedValue = true;
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion

    }
}

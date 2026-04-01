using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public unsafe sealed class FixedBuffer : IDisposable
    {
        public long Length { get; }
        public IntPtr Address => (IntPtr)Pointer;
        public float* Pointer { get; private set; }

        public FixedBuffer(long length)
        {
            Length = length;
            Pointer = (float*)NativeMemory.Alloc((nuint)length, sizeof(float));
        }

        // 直接读写
        public float this[int index]
        {
            get => Pointer[index];
            set => Pointer[index] = value;
        }

        public void Dispose()
        {
            if (Pointer != null)
            {
                NativeMemory.Free(Pointer);
              
                Pointer = null;
            }
        }
    }
}

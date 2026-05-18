using Meloht.WinFormsViewImage;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Management;
using System.Text;
using System.Windows.Forms;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.WinFormsDemo
{
    public partial class FormResult : Form
    {
        private string _model;
        private string _image;
        private Mat _resultImage;
        private string _modelType;

        public FormResult(string model, string image)
        {
            InitializeComponent();
            _model = model;
            _image = image;
        }

        private void FormResult_Load(object sender, EventArgs e)
        {
            int deviceId = Utils.GetMainGPU();
            using var yolo = new YoloSharp(new ExecutionProviderDirectML(_model, deviceId));
            string res = string.Empty;
            _resultImage = Cv2.ImRead(_image);
            _modelType = yolo.CurrentModelType.GetDescription();
            if (yolo.CurrentModelType == ModelType.ObjectDetection)
            {
                var result = yolo.RunDetectWithTime(_image);
                res = $"{result.ToString()}, {result.SpeedResult.ToString()}";

                yolo.DrawDetections(_resultImage, result.Items);
                ShowImageForm(_image, _resultImage.ToBytes());
            }
            else if (yolo.CurrentModelType == ModelType.Classification)
            {
                var result = yolo.RunClassifyWithTime(_image);
                res = $"{result.ToString()}, {result.SpeedResult.ToString()}";

                yolo.DrawClassification(_resultImage, result.Items);
                ShowImageForm(_image, _resultImage.ToBytes());
            }
            else if (yolo.CurrentModelType == ModelType.Segmentation)
            {
                var result = yolo.RunSegmentWithTime(_image);
                res = $"{result.ToString()}, {result.SpeedResult.ToString()}";

                yolo.DrawSegment(_resultImage, result.Items);
                ShowImageForm(_image, _resultImage.ToBytes());
            }


            this.txtReuslt.Text = res;



        }

        private void ShowImageForm(string path, byte[] image)
        {
            var formView = new FormShowImage(path, image);
            formView.TopLevel = false;

            formView.FormBorderStyle = FormBorderStyle.None;

            // 填充整个 GroupBox
            formView.Dock = DockStyle.Fill;

            // 可选：不显示标题栏图标
            formView.ControlBox = false;

            // 添加到 GroupBox
            groupBoxImage.Controls.Clear();
            groupBoxImage.Controls.Add(formView);

            // 显示
            formView.Show();
        }

        private void saveResultImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (_resultImage != null)
            {
                saveFileDialog1.Filter = "JPEG Image (*.jpg)|*.jpg|PNG Image (*.png)|*.png";
                saveFileDialog1.FileName=$"res_{_modelType}_{Path.GetFileName(_image)}";
                if (saveFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    _resultImage.SaveImage(saveFileDialog1.FileName);
                }
            }
        }

        private void FormResult_FormClosing(object sender, FormClosingEventArgs e)
        {
            _resultImage?.Dispose();
        }
    }
}

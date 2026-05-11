namespace YoloSharpOnnx.WinFormsDemo
{
    partial class FormTest
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            label1 = new Label();
            textBoxModel = new TextBox();
            label2 = new Label();
            textBoxImage = new TextBox();
            groupBoxImageView = new GroupBox();
            groupBox2 = new GroupBox();
            btnRunDetect = new Button();
            btnSelectImage = new Button();
            btnSelectModel = new Button();
            openFileDialogModel = new OpenFileDialog();
            openFileDialogImage = new OpenFileDialog();
            groupBox2.SuspendLayout();
            SuspendLayout();
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(12, 24);
            label1.Name = "label1";
            label1.Size = new Size(69, 17);
            label1.TabIndex = 0;
            label1.Text = "Model File";
            // 
            // textBoxModel
            // 
            textBoxModel.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            textBoxModel.Location = new Point(87, 21);
            textBoxModel.Multiline = true;
            textBoxModel.Name = "textBoxModel";
            textBoxModel.ReadOnly = true;
            textBoxModel.Size = new Size(593, 46);
            textBoxModel.TabIndex = 1;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(13, 86);
            label2.Name = "label2";
            label2.Size = new Size(68, 17);
            label2.TabIndex = 2;
            label2.Text = "Image File";
            // 
            // textBoxImage
            // 
            textBoxImage.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            textBoxImage.Location = new Point(87, 83);
            textBoxImage.Multiline = true;
            textBoxImage.Name = "textBoxImage";
            textBoxImage.ReadOnly = true;
            textBoxImage.Size = new Size(593, 46);
            textBoxImage.TabIndex = 3;
            // 
            // groupBoxImageView
            // 
            groupBoxImageView.Dock = DockStyle.Fill;
            groupBoxImageView.Location = new Point(0, 147);
            groupBoxImageView.Name = "groupBoxImageView";
            groupBoxImageView.Size = new Size(803, 330);
            groupBoxImageView.TabIndex = 4;
            groupBoxImageView.TabStop = false;
            groupBoxImageView.Text = "ImageView";
            // 
            // groupBox2
            // 
            groupBox2.Controls.Add(btnRunDetect);
            groupBox2.Controls.Add(btnSelectImage);
            groupBox2.Controls.Add(btnSelectModel);
            groupBox2.Controls.Add(textBoxImage);
            groupBox2.Controls.Add(label2);
            groupBox2.Controls.Add(label1);
            groupBox2.Controls.Add(textBoxModel);
            groupBox2.Dock = DockStyle.Top;
            groupBox2.Location = new Point(0, 0);
            groupBox2.Name = "groupBox2";
            groupBox2.Size = new Size(803, 147);
            groupBox2.TabIndex = 5;
            groupBox2.TabStop = false;
            groupBox2.Text = "Input Data";
            // 
            // btnRunDetect
            // 
            btnRunDetect.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnRunDetect.Location = new Point(692, 103);
            btnRunDetect.Name = "btnRunDetect";
            btnRunDetect.Size = new Size(105, 26);
            btnRunDetect.TabIndex = 6;
            btnRunDetect.Text = "Run Detect";
            btnRunDetect.UseVisualStyleBackColor = true;
            btnRunDetect.Click += btnRun_Click;
            // 
            // btnSelectImage
            // 
            btnSelectImage.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSelectImage.Location = new Point(692, 62);
            btnSelectImage.Name = "btnSelectImage";
            btnSelectImage.Size = new Size(105, 26);
            btnSelectImage.TabIndex = 5;
            btnSelectImage.Text = "SelectImage";
            btnSelectImage.UseVisualStyleBackColor = true;
            btnSelectImage.Click += btnSelectImage_Click;
            // 
            // btnSelectModel
            // 
            btnSelectModel.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSelectModel.Location = new Point(692, 21);
            btnSelectModel.Name = "btnSelectModel";
            btnSelectModel.Size = new Size(105, 26);
            btnSelectModel.TabIndex = 4;
            btnSelectModel.Text = "SelectModel";
            btnSelectModel.UseVisualStyleBackColor = true;
            btnSelectModel.Click += btnSelectModel_Click;
            // 
            // openFileDialogModel
            // 
            openFileDialogModel.FileName = "openFileDialogModel";
            openFileDialogModel.Filter = "onnx model|*.onnx";
            // 
            // openFileDialogImage
            // 
            openFileDialogImage.FileName = "openFileDialogImage";
            openFileDialogImage.Filter = "*.*|*.bmp;*.jpg;*.jpeg;*.tiff;*.tiff;*.png";
            // 
            // FormTest
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(803, 477);
            Controls.Add(groupBoxImageView);
            Controls.Add(groupBox2);
            Name = "FormTest";
            Text = "YoloSharpOnnxFormTest";
            groupBox2.ResumeLayout(false);
            groupBox2.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private Label label1;
        private TextBox textBoxModel;
        private Label label2;
        private TextBox textBoxImage;
        private GroupBox groupBoxImageView;
        private GroupBox groupBox2;
        private Button btnRunDetect;
        private Button btnSelectImage;
        private Button btnSelectModel;
        private OpenFileDialog openFileDialogModel;
        private OpenFileDialog openFileDialogImage;
    }
}

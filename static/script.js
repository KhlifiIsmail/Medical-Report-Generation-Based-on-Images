// Fixed JavaScript for handling the upload response
function uploadImage() {
  const fileInput = document.getElementById("file-input");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first!");
    return;
  }

  // Show loading state
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "<p>Analyzing image... Please wait.</p>";

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Response:", data); // Debug log

      // Handle the response properly based on your backend structure
      if (data.status === "success") {
        resultDiv.innerHTML = `
                <div class="result-success">
                    <h3>Analysis Complete</h3>
                    <p><strong>Result:</strong> ${data.prediction}</p>
                    <p><strong>Confidence:</strong> ${data.confidence}</p>
                </div>
            `;
      } else if (data.error) {
        resultDiv.innerHTML = `
                <div class="result-error">
                    <h3>Analysis Error</h3>
                    <p>${data.error}</p>
                </div>
            `;
      } else {
        // Fallback - display whatever data we got
        resultDiv.innerHTML = `
                <div class="result-info">
                    <h3>Analysis Result</h3>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      resultDiv.innerHTML = `
            <div class="result-error">
                <h3>Upload Failed</h3>
                <p>Error: ${error.message}</p>
            </div>
        `;
    });
}

// Alternative version if your backend returns different structure
function uploadImageAlternative() {
  const fileInput = document.getElementById("file-input");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first!");
    return;
  }

  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "<p>Analyzing image... Please wait.</p>";

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Full response:", data); // Debug

      // Handle multiple possible response formats
      let prediction = "";
      let confidence = "";

      // Try different property names your backend might use
      if (data.prediction) {
        prediction = data.prediction;
      } else if (data.result) {
        prediction = data.result;
      } else if (data.message) {
        prediction = data.message;
      } else if (data.Normal) {
        prediction = "Normal X-ray";
      } else if (data.Pneumonia) {
        prediction = "Pneumonia detected";
      } else {
        prediction = "Analysis completed";
      }

      if (data.confidence) {
        confidence = data.confidence;
      } else if (data.probability) {
        confidence = `${(data.probability * 100).toFixed(1)}%`;
      } else {
        confidence = "N/A";
      }

      resultDiv.innerHTML = `
            <div class="result-success">
                <h3>Analysis Complete</h3>
                <p><strong>Result:</strong> ${prediction}</p>
                <p><strong>Confidence:</strong> ${confidence}</p>
            </div>
        `;
    })
    .catch((error) => {
      console.error("Error:", error);
      resultDiv.innerHTML = `<p class="error">Upload failed: ${error.message}</p>`;
    });
}

// Event listeners
document.addEventListener("DOMContentLoaded", function () {
  const uploadButton = document.getElementById("upload-button");
  if (uploadButton) {
    uploadButton.addEventListener("click", uploadImage);
  }

  // Handle drag and drop if you have it
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        document.getElementById("file-input").files = files;
        uploadImage();
      }
    });
  }
});

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const body = document.body;
    const browseBtn = document.getElementById('browseBtn');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');
    const errorMessage = document.getElementById('errorMessage');
    const successMessage = document.getElementById('successMessage');
    const themeToggle = document.getElementById('themeToggle');

    // Sidebar elements
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const closeSidebar = document.getElementById('closeSidebar');
    const fileList = document.getElementById('fileList');

    // Theme toggle functionality
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('light-mode');
        if (document.body.classList.contains('light-mode')) {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });

    // Sidebar toggle functionality
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('open');
        loadFiles();
    });

    closeSidebar.addEventListener('click', function() {
        sidebar.classList.remove('open');
    });

    // Load files from server
    function loadFiles() {
        fetch('/list_files')
            .then(response => response.json())
            .then(data => {
                if (data.files) {
                    fileList.innerHTML = '';
                    data.files.forEach(file => {
                        const fileItem = document.createElement('li');
                        fileItem.className = 'file-item';
                        fileItem.innerHTML = `
                            <div class="file-info">
                                <div class="file-name" title="${file.name}">${file.name}</div>
                                <div class="file-meta">${file.size} • ${file.date}</div>
                            </div>
                            <button class="file-upload-btn" data-filename="${file.name}" title="Upload ${file.name}" style=" border-radius: 20% 80% 20% 80% / 80% 20% 80% 20%;"">
                                <i class="fas fa-upload"></i> Upload
                            </button>
                        `;
                        fileList.appendChild(fileItem);
                    });

                    // Add event listeners to upload buttons
                    document.querySelectorAll('.file-upload-btn').forEach(btn => {
                        btn.addEventListener('click', function() {
                            const filename = this.getAttribute('data-filename');
                            uploadSelectedFile(filename);
                        });
                    });
                }
            })
            .catch(error => {
                console.error('Error loading files:', error);
            });
    }

    // Upload selected file from sidebar
    function uploadSelectedFile(filename) {
        errorMessage.style.display = 'none';
        successMessage.style.display = 'none';
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        progressStatus.textContent = 'Uploading...';

        // Start the progress animation
        const startTime = Date.now();
        const totalDuration = 5500; // 15 seconds
        const redirectAtPercentage = 0.8; // Redirect at 60%
        let redirectTriggered = false;

        const progressInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / totalDuration, 1);
            const percent = Math.round(progress * 100);

            progressBar.style.width = percent + '%';
            progressPercent.textContent = percent + '%';

            // Redirect at 60% progress (9 seconds)
            if (progress >= redirectAtPercentage && !redirectTriggered) {
                redirectTriggered = true;
                progressStatus.textContent = 'Processing... Redirecting now';

                // Start the actual upload in background
                fetch('/upload', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `selected_file=${encodeURIComponent(filename)}`
                }).then(response => {
                    if (!response.ok) {
                        console.error('Background upload failed');
                    }
                }).catch(error => {
                    console.error('Background upload error:', error);
                });

                // Redirect while animation continues
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 500);
            }

            if (progress >= 1) {
                clearInterval(progressInterval);
            }
        }, 100);
    }

    // Handle file selection
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('active');
    }

    function unhighlight() {
        uploadArea.classList.remove('active');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFile(files[0]);
        }
    }

    function handleFileSelect() {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    }

    function handleFile(file) {
        errorMessage.style.display = 'none';
        successMessage.style.display = 'none';

        // Validate file type
        if (!/\.(csv|xls|xlsx)$/i.test(file.name)) {
            showError('Please upload a valid CSV or Excel file');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size exceeds 10MB limit');
            return;
        }

        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        progressStatus.textContent = 'Uploading...';

        // Upload the file with 15-second progress
        uploadFile(file);
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        const startTime = Date.now();
        const totalDuration = 5500; // 15 seconds
        const redirectAtPercentage = 0.8; // Redirect at 60%
        let redirectTriggered = false;

        // Progress animation
        const progressInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / totalDuration, 1);
            const percent = Math.round(progress * 100);

            progressBar.style.width = percent + '%';
            progressPercent.textContent = percent + '%';

            // Redirect at 60% progress (9 seconds)
            if (progress >= redirectAtPercentage && !redirectTriggered) {
                redirectTriggered = true;
                progressStatus.textContent = 'Processing... Redirecting now';

                // Redirect while animation continues
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 500);
            }

            if (progress >= 1) {
                clearInterval(progressInterval);
            }
        }, 100);

        xhr.upload.addEventListener('progress', function(e) {
            // Actual upload progress (not shown to user)
            if (e.lengthComputable) {
                console.log('Actual upload progress:', Math.round((e.loaded / e.total) * 100) + '%');
            }
        });

        xhr.addEventListener('load', function() {
            clearInterval(progressInterval);
            if (xhr.status !== 200 && !redirectTriggered) {
                showError('Upload failed: ' + xhr.statusText);
                resetUpload();
            }
        });

        xhr.addEventListener('error', function() {
            clearInterval(progressInterval);
            if (!redirectTriggered) {
                showError('Upload failed - network error');
                resetUpload();
            }
        });

        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        successMessage.style.display = 'none';
    }

    function showSuccess(message) {
        successMessage.textContent = message;
        successMessage.style.display = 'block';
        errorMessage.style.display = 'none';
    }

    function resetUpload() {
        progressContainer.style.display = 'none';
        fileInput.value = '';
    }

    // Easing function for smooth progress animation
    function easeInOut(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
});
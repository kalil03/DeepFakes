document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const uploadContent = document.querySelector('.upload-content');

    let currentFile = null;

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
                analyzeBtn.disabled = false;
                resultSection.style.display = 'none';
            } else {
                alert('Por favor, selecione apenas arquivos de imagem.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            imagePreview.src = reader.result;
            previewContainer.style.display = 'block';
            uploadContent.style.display = 'none';
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadContent.style.display = 'flex';
        analyzeBtn.disabled = true;
        resultSection.style.display = 'none';
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        const btnText = analyzeBtn.querySelector('.btn-text');
        const loader = analyzeBtn.querySelector('.loader');

        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            const endTime = performance.now();
            const processTime = (endTime - startTime).toFixed(0);

            if (response.ok) {
                showResult(data, processTime);
            } else {
                alert('Erro na análise: ' + (data.error || 'Erro desconhecido'));
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Erro ao conectar com o servidor.');
        } finally {
            // Reset UI State
            btnText.style.display = 'inline';
            loader.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    function showResult(data, time) {
        resultSection.style.display = 'block';

        const badge = document.getElementById('result-badge');
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceBar = document.getElementById('confidence-bar');
        const processTime = document.getElementById('process-time');

        // Configurar Badge
        // Assumindo que a classe '1' é Real e '0' é Fake (ou vice-versa, ajustaremos se necessário)
        // No script original: ImageFolder ordena classes. Se 'real' e 'fake' forem as pastas:
        // fake -> 0, real -> 1.
        // Vamos assumir isso. Se o modelo retornar '0', é Fake.

        // Ajuste baseado no output do modelo:
        // O modelo retorna prediction como string (ex: "0" ou "1")

        let isReal = data.prediction == "1";
        // Se as classes forem ['fake', 'real'], então índice 1 é Real.
        // Se as classes forem ['real', 'fake'], então índice 0 é Real.
        // Vamos verificar data.classes se disponível.

        if (data.classes && data.classes.length === 2) {
            // Tenta inferir pelo nome
            const class0 = String(data.classes[0]).toLowerCase();
            const class1 = String(data.classes[1]).toLowerCase();

            if (class1.includes('real')) {
                isReal = data.prediction == "1";
            } else if (class0.includes('real')) {
                isReal = data.prediction == "0";
            }
        }

        if (isReal) {
            badge.textContent = 'REAL';
            badge.className = 'badge real';
        } else {
            badge.textContent = 'FAKE';
            badge.className = 'badge fake';
        }

        const confidencePercent = (data.confidence * 100).toFixed(1) + '%';
        confidenceValue.textContent = confidencePercent;
        confidenceBar.style.width = confidencePercent;
        processTime.textContent = time + ' ms';

        // Scroll suave até o resultado
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});

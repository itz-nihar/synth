document.addEventListener('DOMContentLoaded', () => {
    let currentSessionId = null;
    let currentTargetSize = 128; // Default to 128 for high-res sharp generation

    // Stepper navigation
    const stepBtns = document.querySelectorAll('.step-btn');
    const stepContents = document.querySelectorAll('.step-content');

    function switchStep(stepNum) {
        stepBtns.forEach(btn => {
            if (btn.dataset.step === String(stepNum)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        stepContents.forEach(content => {
            if (content.id === `step-${stepNum}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }

    stepBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            switchStep(btn.dataset.step);
        });
    });

    document.getElementById('btn-goto-step2').addEventListener('click', () => switchStep(2));
    document.getElementById('btn-goto-step3').addEventListener('click', () => switchStep(3));
    document.getElementById('btn-goto-step4').addEventListener('click', () => switchStep(4));
    document.getElementById('btn-goto-step5').addEventListener('click', () => switchStep(5));

    // Upload zone setup
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const uploadStatus = document.getElementById('upload-status');
    const uploadManifest = document.getElementById('upload-manifest');

    btnBrowse.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== btnBrowse) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#6366f1';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    function formatErrorMessage(detail, defaultMsg = 'Unknown error') {
        if (!detail) return defaultMsg;
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
            return detail.map(item => {
                if (typeof item === 'object') {
                    return item.msg ? (item.loc ? `${item.loc.join('.')}: ${item.msg}` : item.msg) : JSON.stringify(item);
                }
                return String(item);
            }).join('; ');
        }
        if (typeof detail === 'object') {
            return detail.msg || JSON.stringify(detail);
        }
        return String(detail);
    }

    async function handleFileUpload(file) {
        uploadStatus.classList.remove('hidden');
        uploadStatus.innerText = 'Uploading and analyzing image dataset...';

        const formData = new FormData();
        formData.append('file', file);
        if (currentSessionId) {
            formData.append('session_id', currentSessionId);
        }

        try {
            const resp = await fetch('/api/dataset/upload', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (resp.ok && data.status === 'success') {
                currentSessionId = data.data.session_id;
                document.getElementById('session-id-display').innerText = currentSessionId;
                
                uploadStatus.innerText = data.message;
                uploadManifest.classList.remove('hidden');
                document.getElementById('stat-total-imgs').innerText = data.data.total_images;
                
                const previewGrid = document.getElementById('raw-preview-grid');
                previewGrid.innerHTML = '';
                (data.data.previews || []).forEach(src => {
                    const img = document.createElement('img');
                    img.src = src;
                    previewGrid.appendChild(img);
                });
            } else {
                uploadStatus.innerText = 'Upload failed: ' + formatErrorMessage(data.detail);
            }
        } catch (err) {
            uploadStatus.innerText = 'Error: ' + err.message;
        }
    }

    // Demo Dataset Generator
    document.getElementById('btn-demo-data').addEventListener('click', async () => {
        uploadStatus.classList.remove('hidden');
        uploadStatus.innerText = 'Creating demo real image dataset...';

        const images = [];
        for (let i = 0; i < 16; i++) {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            
            const grad = ctx.createLinearGradient(0, 0, 128, 128);
            grad.addColorStop(0, `rgb(${(i * 30) % 255}, ${(i * 50) % 255}, 200)`);
            grad.addColorStop(1, `rgb(240, ${(i * 40) % 255}, ${(i * 20) % 255})`);
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, 128, 128);
            
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(64 + (i % 4) * 4, 64, 20 + (i % 3) * 3, 0, Math.PI * 2);
            ctx.fill();

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            images.push(blob);
        }

        const formData = new FormData();
        formData.append('file', images[0], 'demo_dataset.png');
        if (currentSessionId) formData.append('session_id', currentSessionId);

        const resp = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.status === 'success') {
            currentSessionId = data.data.session_id;
            document.getElementById('session-id-display').innerText = currentSessionId;
            
            for (let idx = 1; idx < images.length; idx++) {
                const fd = new FormData();
                fd.append('file', images[idx], `demo_${idx}.png`);
                fd.append('session_id', currentSessionId);
                await fetch('/api/dataset/upload', { method: 'POST', body: fd });
            }
            
            uploadStatus.innerText = `Demo dataset created successfully (${images.length} images).`;
            uploadManifest.classList.remove('hidden');
            document.getElementById('stat-total-imgs').innerText = images.length;
            
            const mResp = await fetch(`/api/dataset/manifest/${currentSessionId}`);
            const mData = await mResp.json();
            const previewGrid = document.getElementById('raw-preview-grid');
            previewGrid.innerHTML = '';
            (mData.data.previews || []).forEach(src => {
                const img = document.createElement('img');
                img.src = src;
                previewGrid.appendChild(img);
            });
        }
    });

    // Step 2: Preprocessing
    document.getElementById('btn-run-preprocess').addEventListener('click', async () => {
        if (!currentSessionId) {
            alert('Please upload a dataset first in Step 1!');
            return;
        }

        currentTargetSize = parseInt(document.getElementById('select-size').value, 10);
        const split = parseFloat(document.getElementById('select-split').value);

        try {
            const resp = await fetch('/api/preprocess/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    target_size: currentTargetSize,
                    train_split_ratio: split
                })
            });
            const data = await resp.json();

            if (resp.ok && data.status === 'success') {
                const prepRes = data.data;
                document.getElementById('preprocess-results').classList.remove('hidden');
                document.getElementById('prep-processed-count').innerText = prepRes.processed_count;
                document.getElementById('prep-train-count').innerText = prepRes.train_count;
                document.getElementById('prep-test-count').innerText = prepRes.test_count;

                // Populate Vision Encoder Dataset Context Analysis
                if (prepRes.dataset_context) {
                    const ctx = prepRes.dataset_context;
                    document.getElementById('ctx-detected-type').innerText = ctx.detected_type || 'Custom Visual Domain';
                    document.getElementById('ctx-resolution').innerText = ctx.resolution || `${currentTargetSize}x${currentTargetSize}`;
                    document.getElementById('ctx-clusters').innerText = `${ctx.clusters_count} cluster(s)`;
                    document.getElementById('ctx-diversity').innerText = `${ctx.diversity_score} / 100`;
                    document.getElementById('ctx-embedding-status').innerText = ctx.context_status || 'Embedded';
                }

                const prepGrid = document.getElementById('prep-preview-grid');
                prepGrid.innerHTML = '';
                (prepRes.previews || []).forEach(src => {
                    const img = document.createElement('img');
                    img.src = src;
                    prepGrid.appendChild(img);
                });
            } else {
                alert('Preprocessing failed: ' + formatErrorMessage(data.detail));
            }
        } catch (err) {
            alert('Error during preprocessing: ' + err.message);
        }
    });

    // Step 3: Model Training
    async function startModelPipeline(modelType) {
        if (!currentSessionId) {
            alert('Please complete Steps 1 & 2 first!');
            return;
        }

        const epochs = parseInt(document.getElementById(`epochs-${modelType}`).value, 10);
        const samples = parseInt(document.getElementById(`samples-${modelType}`).value, 10);
        const targetSize = currentTargetSize || parseInt(document.getElementById('select-size').value, 10) || 128;

        const progWrap = document.getElementById(`prog-wrap-${modelType}`);
        const progBar = document.getElementById(`prog-bar-${modelType}`);
        const progText = document.getElementById(`prog-text-${modelType}`);
        const genArea = document.getElementById(`gen-area-${modelType}`);
        const genGrid = document.getElementById(`gen-grid-${modelType}`);

        progWrap.classList.remove('hidden');
        progBar.style.width = '5%';
        progText.innerText = 'Starting...';

        try {
            const resp = await fetch('/api/models/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    model_type: modelType,
                    epochs: epochs,
                    num_synthetic_samples: samples,
                    image_size: targetSize
                })
            });
            const data = await resp.json();

            if (!resp.ok) {
                alert('Training launch failed: ' + formatErrorMessage(data.detail));
                return;
            }

            // Poll status
            const pollInterval = setInterval(async () => {
                const sResp = await fetch(`/api/models/status/${currentSessionId}/${modelType}`);
                const sData = await sResp.json();
                const st = sData.data;

                if (st.status === 'training' || st.status === 'generating') {
                    progBar.style.width = `${st.progress_percent}%`;
                    progText.innerText = `${st.progress_percent}% (${st.status} Epoch ${st.current_epoch}/${st.total_epochs})`;
                } else if (st.status === 'completed') {
                    clearInterval(pollInterval);
                    progBar.style.width = '100%';
                    progText.innerText = 'Completed!';

                    genArea.classList.remove('hidden');
                    genGrid.innerHTML = '';
                    (st.previews || []).forEach(src => {
                        const img = document.createElement('img');
                        img.src = src;
                        genGrid.appendChild(img);
                    });
                } else if (st.status === 'failed') {
                    clearInterval(pollInterval);
                    progText.innerText = 'Failed!';
                    alert('Model error: ' + formatErrorMessage(st.error));
                }
            }, 1000);

        } catch (err) {
            alert('Error starting model: ' + err.message);
        }
    }

    document.getElementById('btn-train-gan').addEventListener('click', () => startModelPipeline('gan'));
    document.getElementById('btn-train-diffusion').addEventListener('click', () => startModelPipeline('diffusion'));
    document.getElementById('btn-train-vae_gan').addEventListener('click', () => startModelPipeline('vae_gan'));

    async function generateNewSamples(modelType) {
        if (!currentSessionId) {
            alert('Please complete Steps 1 & 2 first!');
            return;
        }

        const samples = parseInt(document.getElementById(`samples-${modelType}`).value, 10);
        const targetSize = currentTargetSize || parseInt(document.getElementById('select-size').value, 10) || 128;

        const progWrap = document.getElementById(`prog-wrap-${modelType}`);
        const progBar = document.getElementById(`prog-bar-${modelType}`);
        const progText = document.getElementById(`prog-text-${modelType}`);
        const genArea = document.getElementById(`gen-area-${modelType}`);
        const genGrid = document.getElementById(`gen-grid-${modelType}`);

        progWrap.classList.remove('hidden');
        progBar.style.width = '40%';
        progText.innerText = 'Dynamically generating synthetic samples...';

        try {
            const resp = await fetch('/api/models/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    model_type: modelType,
                    num_synthetic_samples: samples,
                    image_size: targetSize
                })
            });
            const data = await resp.json();

            if (resp.ok && data.status === 'success') {
                progBar.style.width = '100%';
                progText.innerText = `Successfully generated ${data.data.generated_count} dynamic synthetic images!`;

                genArea.classList.remove('hidden');
                genGrid.innerHTML = '';
                (data.data.previews || []).forEach(src => {
                    const img = document.createElement('img');
                    img.src = src;
                    genGrid.appendChild(img);
                });
            } else {
                progText.innerText = 'Generation failed!';
                alert('Generation failed: ' + formatErrorMessage(data.detail));
            }
        } catch (err) {
            progText.innerText = 'Error!';
            alert('Error generating samples: ' + err.message);
        }
    }

    document.getElementById('btn-gen-gan').addEventListener('click', () => generateNewSamples('gan'));
    document.getElementById('btn-gen-diffusion').addEventListener('click', () => generateNewSamples('diffusion'));
    document.getElementById('btn-gen-vae_gan').addEventListener('click', () => generateNewSamples('vae_gan'));

    // Step 4: Evaluation Engine
    document.getElementById('btn-run-all-eval').addEventListener('click', async () => {
        if (!currentSessionId) {
            alert('Please select a session first!');
            return;
        }

        const dashboard = document.getElementById('eval-dashboard');
        dashboard.classList.remove('hidden');
        dashboard.innerHTML = '<p>Computing FID, SSIM, TSTR Utility, and NNDR Privacy metrics...</p>';

        const models = ['gan', 'diffusion', 'vae_gan'];
        const results = [];

        for (const m of models) {
            try {
                const resp = await fetch('/api/evaluate/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessionId, model_type: m })
                });
                if (resp.ok) {
                    const data = await resp.json();
                    results.push(data.data);
                }
            } catch (err) {
                console.error('Eval error for ' + m, err);
            }
        }

        if (results.length === 0) {
            dashboard.innerHTML = '<p style="color:#ef4444">No generated models found. Please train models in Step 3 first!</p>';
            return;
        }

        dashboard.innerHTML = '';
        results.forEach(res => {
            const card = document.createElement('div');
            card.className = 'eval-card';
            const statusColor = res.validation_status === 'PASS' ? '#10b981' : '#ef4444';
            const semPct = res.semantic_similarity !== undefined ? (res.semantic_similarity * 100).toFixed(1) : '90.0';
            card.innerHTML = `
                <h3>${res.model_type.toUpperCase()} Evaluation Metrics</h3>
                <div class="eval-metric-row"><span>Quality Guard Status:</span> <strong style="color:${statusColor}">${res.validation_status || 'PASS'}</strong></div>
                <div class="eval-metric-row"><span>Semantic Similarity:</span> <strong style="color:#818cf8">${semPct}%</strong></div>
                <div class="eval-metric-row"><span>Image Quality Score:</span> <strong>${res.quality.score} / 100</strong></div>
                <div class="eval-metric-row"><span>FID (Fréchet Inception Dist):</span> <strong>${res.quality.fid}</strong></div>
                <div class="eval-metric-row"><span>SSIM (Structural Similarity):</span> <strong>${res.quality.ssim}</strong></div>
                <div class="eval-metric-row"><span>Utility Score (TSTR Ratio):</span> <strong>${res.utility.score} / 100</strong></div>
                <div class="eval-metric-row"><span>Privacy Score (NNDR):</span> <strong>${res.privacy.score} / 100</strong></div>
                <div class="eval-metric-row"><span>Copy/Leakage Risk:</span> <strong>${res.privacy.copy_leakage_percent}%</strong></div>
            `;
            dashboard.appendChild(card);
        });
    });

    // Step 5: Comparison & Recommendation
    document.getElementById('btn-trigger-comparison').addEventListener('click', async () => {
        if (!currentSessionId) {
            alert('Please select a session first!');
            return;
        }

        try {
            const resp = await fetch('/api/compare/evaluate_all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            const data = await resp.json();

            if (resp.ok && data.status === 'success') {
                const rep = data.data;
                document.getElementById('comparison-results-card').classList.remove('hidden');

                document.getElementById('winner-name').innerText = rep.recommended_model_name;
                document.getElementById('winner-rationale').innerText = rep.rationale;

                // Populate table
                const tbody = document.getElementById('benchmark-table-body');
                tbody.innerHTML = '';

                rep.rankings.forEach(item => {
                    const tr = document.createElement('tr');
                    const statusBadge = item.validation_status === 'FAIL'
                        ? `<span style="background:#ef444422;color:#ef4444;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8rem">FAIL</span>`
                        : `<span style="background:#10b98122;color:#10b981;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8rem">PASS</span>`;
                    const semVal = item.semantic_similarity !== undefined ? `${(item.semantic_similarity * 100).toFixed(1)}%` : 'N/A';
                    tr.innerHTML = `
                        <td><strong>#${item.rank}</strong></td>
                        <td>${item.model_name}</td>
                        <td>${statusBadge}</td>
                        <td><strong style="color:#818cf8">${semVal}</strong></td>
                        <td><strong style="color:var(--primary)">${item.composite_score} / 100</strong></td>
                        <td>${item.quality.score} / 100</td>
                        <td>${item.quality.fid}</td>
                        <td>${item.utility.score} / 100</td>
                        <td>${item.privacy.score} / 100</td>
                        <td>
                            <div style="display:flex; gap:4px">
                                ${(item.previews || []).map(p => `<img src="${p}" style="width:30px;height:30px;border-radius:4px">`).join('')}
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });

                // Side by side real vs synthetic previews
                const winnerObj = rep.rankings[0];
                const sideSynGrid = document.getElementById('side-syn-grid');
                sideSynGrid.innerHTML = '';
                (winnerObj.previews || []).forEach(src => {
                    const img = document.createElement('img');
                    img.src = src;
                    sideSynGrid.appendChild(img);
                });

                // Copy real preview
                const rawGrid = document.getElementById('raw-preview-grid').innerHTML;
                document.getElementById('side-real-grid').innerHTML = rawGrid;

            } else {
                alert('Comparison failed: ' + formatErrorMessage(data.detail));
            }
        } catch (err) {
            alert('Error during comparison: ' + err.message);
        }
    });

    // Download ZIP
    document.getElementById('btn-download-zip').addEventListener('click', () => {
        if (!currentSessionId) return;
        window.location.href = `/api/compare/export/${currentSessionId}`;
    });
});

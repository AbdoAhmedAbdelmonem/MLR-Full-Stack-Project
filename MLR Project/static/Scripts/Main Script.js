function toggleCode(button) {
    const plotCard = button.closest('.plot-card');
    const codePanel = plotCard.querySelector('.code-panel');

    document.querySelectorAll('.code-panel').forEach(panel => {
        if (panel !== codePanel) {
            panel.classList.remove('active');
            panel.closest('.plot-card').classList.remove('code-active');
        }
    });

    plotCard.classList.toggle('code-active');
    codePanel.classList.toggle('active');

    button.textContent = codePanel.classList.contains('active') ? 'Hide Code' : 'Show Code';

    setTimeout(() => {
        if (window.Plotly) {
            const plotDiv = plotCard.querySelector('.plotly-graph-div');
            if (plotDiv) Plotly.Plots.resize(plotDiv);
        }
    }, 400);
}

function openTab(evt, tabName) {
    // Hide all tab contents
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    document.querySelectorAll('.code-panel').forEach(panel => {
        panel.classList.remove('active');
        panel.closest('.plot-card').classList.remove('code-active');
        panel.previousElementSibling.textContent = 'Show Code';
    });

    document.getElementById(tabName).classList.add('active');

    handleResize();
}
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                const intro = document.getElementById('introOverlay');
                intro.style.opacity = '0';

                setTimeout(function() {
                    intro.remove();
                }, 1500);
            }, 3000);

        });
        let resizeTimeout;
        function handleResize() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (window.Plotly) {
                    const plots = document.querySelectorAll('.plotly-graph-div');
                    plots.forEach(plot => {
                        Plotly.Plots.resize(plot);
                    });
                }
            }, 100);
        }

        document.addEventListener('DOMContentLoaded', function() {
            document.querySelector('.tab-content').classList.add('active');

            setTimeout(handleResize, 300);

            const sidebar = document.querySelector('.sidebar');
            sidebar.addEventListener('transitionend', function(e) {
                if (e.propertyName === 'width') {
                    handleResize();
                }
            });

            window.addEventListener('resize', handleResize);
        });

        function openTab(evt, tabName) {
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }

            document.getElementById(tabName).classList.add('active');

            handleResize();
        }

        document.querySelectorAll('.plot-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px)';
                this.style.boxShadow = '0 12px 40px rgba(187, 134, 252, 0.2)';
                this.style.zIndex = '10';
            });

            card.addEventListener('mouseleave', function() {
                this.style.transform = '';
                this.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
                this.style.zIndex = '';
            });
        });

        const sidebar = document.querySelector('.sidebar');
        sidebar.addEventListener('mousemove', (e) => {
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;
            sidebar.style.transform = `perspective(1000px) rotateY(${(x - 0.5) * 5}deg) rotateX(${(0.5 - y) * 5}deg)`;
        });

        sidebar.addEventListener('mouseleave', () => {
            sidebar.style.transform = 'perspective(1000px) rotateY(0deg) rotateX(0deg)';
        });
        // Add this to your existing script
        function animateCounters() {
            const counters = document.querySelectorAll('.counter');
            const speed = 200; // Lower is faster

            counters.forEach(counter => {
                const target = +counter.getAttribute('data-target');
                const count = +counter.innerText;
                const increment = target / speed;

                if (count < target) {
                    counter.innerText = Math.ceil(count + increment);
                    setTimeout(animateCounters, 1);
                } else {
                    counter.innerText = target;
                }
            });
        }

        // Call this when the cleaning tab is shown
        document.addEventListener('DOMContentLoaded', function() {
            // Start counters when cleaning tab is active
            const cleaningTab = document.getElementById('cleaning-tab');
            if (cleaningTab) {
                cleaningTab.addEventListener('click', function() {
                    setTimeout(animateCounters, 500);
                });
            }

            // Also start if cleaning tab is active by default
            if (document.getElementById('cleaning').classList.contains('active')) {
                setTimeout(animateCounters, 1000);
            }
        });
  let currentlyActiveButton = null;

        function handleButtonClick(button, estimator) {
    // Remove active class from previously active button
    if (currentlyActiveButton) {
      currentlyActiveButton.classList.remove('active');
    }

    // Add active class to clicked button
    button.classList.add('active');
    currentlyActiveButton = button;

    // Call your original function
    showEstimatorResults(estimator);
  }
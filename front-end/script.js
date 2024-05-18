document.addEventListener('DOMContentLoaded', function() {
    populateLearningBiases();

    document.getElementById('aNail').addEventListener('click', function() {
        choosePath('A-nail');
    });
    document.getElementById('bNail').addEventListener('click', function() {
        choosePath('B-nail');
    });
    document.getElementById('cNail').addEventListener('click', function() {
        choosePath('C-nail');
    });
});

function populateLearningBiases() {
    const biases = ['ε_ref', 'ε_stat', 'ε_apx', 'ε_opt'];
    const container = document.getElementById('biasOptions');

    biases.forEach(bias => {
        let button = document.createElement('button');
        button.textContent = bias;
        container.appendChild(button);
    });
}

function choosePath(path) {
    alert(`You have chosen the ${path} path.`);
}

const barsElement = document.querySelector('#bars');
const shuffleButton = document.querySelector('#shuffleButton');
const runButton = document.querySelector('#runButton');
const themeToggle = document.querySelector('#themeToggle');
const arrayLabel = document.querySelector('#arrayLabel');
let values = [42, 76, 34, 88, 51, 64, 25, 92, 57, 70, 39, 81, 47, 61];
let selectedSort = 'bubble';
let isSorting = false;

function drawBars(activeIndex = -1) {
  barsElement.innerHTML = '';
  values.forEach((value, index) => {
    const bar = document.createElement('span');
    bar.className = `bar${index === activeIndex ? ' active' : ''}`;
    bar.style.height = `${value}%`;
    bar.setAttribute('aria-label', `Value ${value}`);
    barsElement.appendChild(bar);
  });
}

function randomize() {
  values = values.map(() => Math.floor(Math.random() * 72) + 20);
  arrayLabel.textContent = 'RANDOM';
  drawBars();
}

function wait(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function bubbleSort() {
  for (let end = values.length - 1; end > 0; end -= 1) {
    for (let index = 0; index < end; index += 1) {
      drawBars(index);
      await wait(55);
      if (values[index] > values[index + 1]) {
        [values[index], values[index + 1]] = [values[index + 1], values[index]];
      }
    }
  }
}

async function selectionSort() {
  for (let start = 0; start < values.length - 1; start += 1) {
    let smallest = start;
    for (let index = start + 1; index < values.length; index += 1) {
      drawBars(index);
      await wait(50);
      if (values[index] < values[smallest]) smallest = index;
    }
    [values[start], values[smallest]] = [values[smallest], values[start]];
  }
}

async function runSort() {
  if (isSorting) return;
  isSorting = true;
  runButton.disabled = true;
  runButton.innerHTML = 'Sorting… <span>↻</span>';
  if (selectedSort === 'bubble') await bubbleSort();
  else await selectionSort();
  drawBars();
  arrayLabel.textContent = 'SORTED';
  runButton.disabled = false;
  runButton.innerHTML = 'Run sort <span>▶</span>';
  isSorting = false;
}

document.querySelectorAll('.filter').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.filter').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    const filter = button.dataset.filter;
    document.querySelectorAll('.path-card').forEach(card => {
      card.hidden = filter !== 'all' && card.dataset.category !== filter;
    });
  });
});

document.querySelectorAll('.sort-button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.sort-button').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    selectedSort = button.dataset.sort;
  });
});

shuffleButton.addEventListener('click', randomize);
runButton.addEventListener('click', runSort);
themeToggle.addEventListener('click', () => document.body.classList.toggle('dark'));
drawBars();

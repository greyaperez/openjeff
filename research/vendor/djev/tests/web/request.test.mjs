import test from 'node:test';
import assert from 'node:assert/strict';
import { validateRequest, requestCode, createLiveLoop, isNativeImage } from '../../playground/src/request.js';

const fixture = () => ({ model: 'djev', state: { message: 'Help with checkout.' }, questions: {
  urgent: { type: 'noul', instructions: 'Is this urgent?' },
  category: { type: 'choice', criteria: { a: 'Technical', b: 'Billing' } },
  impact: { type: 'score', criteria: ['Minor', 'Major'] }
} });
const image = 'data:image/png;base64,aGVsbG8=';

test('only supported inline images can render a preview; image-named JSON remains ordinary data', () => {
  assert.equal(isNativeImage({ image, text: 'Inline' }), true);
  for (const value of [{ image: 'https://example.invalid/track.png' }, { image: '//example.invalid/track.png' },
    { image: 'data:image/svg+xml;base64,PHN2Zz4=' }, { image: 'javascript:alert(1)' },
    { image, extra: true }, { image, text: {} }, { image: '' }]) assert.equal(isNativeImage(value), false);
  const request = fixture(); request.questions.urgent.instructions = { image: 'https://example.invalid/ordinary-data' };
  assert.equal(validateRequest(request), request);
});

test('validates structured descriptions and counts images across every question', () => {
  const request = fixture(); request.images = [image];
  request.questions.category.instructions = { image, text: 'Compare.' };
  request.questions.category.criteria = { a: { image }, b: { image, text: 'Alternative' } };
  request.questions.impact.criteria = [{ image }, { image }];
  assert.equal(validateRequest(request), request);
  request.questions.urgent.instructions = { image };
  assert.throws(() => validateRequest(request), /at most 6/);
});

test('rejects malformed question contracts, missing image options and oversized state', () => {
  for (const questions of [[], {}, { x: { type: 'other' } }, { x: { type: 'score', criteria: ['One'] } },
    { x: { type: 'choice', criteria: { a: { image: '' } } } }, { x: { type: 'noul', extra: true } }]) {
    assert.throws(() => validateRequest({ state: '', questions }));
  }
  assert.throws(() => validateRequest({ ...fixture(), state: 'a'.repeat(20_001) }), /20,000/);
  assert.throws(() => validateRequest({ ...fixture(), state: null }), /State must/);
  const request = fixture(); request.questions.impact.criteria[0] = 'a'.repeat(501);
  assert.throws(() => validateRequest(request), /500/);
});

test('exports the exact payload without credentials and quotes shell metacharacters', () => {
  const request = fixture(); request.state = "don't run $(touch /tmp/unwanted) `anything`\nquoted \"text\"";
  assert.deepEqual(JSON.parse(requestCode(request, 'json')), request);
  const curl = requestCode(request, 'curl');
  assert.ok(curl.includes("don'\\''t"));
  assert.ok(curl.includes('http://127.0.0.1:8000/v1/request'));
  for (const language of ['curl', 'python', 'typescript']) {
    assert.ok(!requestCode(request, language).includes('Authorization'));
  }
  const python = requestCode(request, 'python');
  const literal = python.match(/payload = json.loads\((.*)\)/)[1];
  assert.deepEqual(JSON.parse(JSON.parse(literal)), request);
});

test('live loop awaits each request and stopping prevents a late response from scheduling another', async () => {
  let runs = 0, release; const errors = [];
  const loop = createLiveLoop(async () => { runs++; await new Promise(resolve => { release = resolve; }); }, error => errors.push(error), 1);
  loop.start(); loop.start();
  await new Promise(resolve => setTimeout(resolve, 12));
  assert.equal(runs, 1); loop.stop(); release();
  await new Promise(resolve => setTimeout(resolve, 12));
  assert.equal(runs, 1); assert.equal(loop.active, false); assert.equal(errors.length, 0);
});

test('live inference errors stop the loop instead of repeatedly sending bad frames', async () => {
  let runs = 0; const errors = [];
  const loop = createLiveLoop(async () => { runs++; throw new Error('API unavailable'); }, error => errors.push(error), 1);
  loop.start(); await new Promise(resolve => setTimeout(resolve, 12));
  assert.equal(runs, 1); assert.equal(errors[0].message, 'API unavailable'); assert.equal(loop.active, false);
});

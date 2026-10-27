// THROWAWAY Showdown spike S4: does the sim replay identically under a fixed seed? Run with the scratch Node from ~/xp-showdown-spike (see the plan).
const fs = require('fs');
const root = process.env.HOME + '/xp-showdown-spike/pokemon-showdown';
const {BattleStream, getPlayerStreams, Teams} = require(root + '/dist/sim');
const fmt = 'gen9championsvgc2026regmc';
const team = n => Teams.pack(Teams.import(fs.readFileSync(process.env.HOME + `/xp-showdown-spike/teams/${n}.txt`, 'utf8')));
async function run(seed) {
  const streams = getPlayerStreams(new BattleStream());
  const log = [];
  (async () => { for await (const c of streams.omniscient) log.push(c); })();
  streams.omniscient.write(`>start ${JSON.stringify({formatid: fmt, seed})}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({name: 'A', team: team('tune1')})}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({name: 'B', team: team('eval1')})}`);
  streams.omniscient.write('>p1 team 1234'); streams.omniscient.write('>p2 team 1234');
  for (let i = 0; i < 6; i++) { streams.omniscient.write('>p1 move 1, move 1'); streams.omniscient.write('>p2 move 1, move 1'); }
  await new Promise(r => setTimeout(r, 1500));
  return log.join('\n').split('\n').filter(l => !l.startsWith('|t:|')).join('\n');
}
(async () => {
  const s1 = [1, 2, 3, 4], s2 = [9, 9, 9, 9];
  const a = await run(s1), b = await run(s1), c = await run(s2);
  console.log('same seed identical:', a === b, '| different seed identical:', a === c, '| log chars', a.length);
  process.exit(0);
})();

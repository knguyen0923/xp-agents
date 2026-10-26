// THROWAWAY: run from the scratch dir with the Showdown checkout at ./pokemon-showdown and missing.json = {"species":[...],"moves":[...]}; writes patch_data.json. See scripts/spikes/showdown_patch_poke_env.py.
const {Dex}=require('./pokemon-showdown/dist/sim');
const fs=require('fs');
const d=Dex.mod('champions');
const spKeys=['abilities','baseForme','baseSpecies','baseStats','battleOnly','canGigantamax','canHatch','cannotDynamax','changesFrom','color','cosmeticFormes','eggGroups','evoCondition','evoItem','evoLevel','evoMove','evoRegion','evoType','evos','forme','formeOrder','gen','gender','genderRatio','heightm','maxHP','mother','name','num','otherFormes','prevo','requiredAbility','requiredItem','requiredItems','requiredMove','requiredTeraType','tags','types','weightkg'];
const mvKeys=['accuracy','basePower','category','contestType','flags','name','num','pp','priority','secondary','target','type','drain','heal','recoil','status','boosts','self','volatileStatus','sideCondition','weather','terrain','multihit','critRatio','willCrit','ohko','isZ','isMax','breaksProtect','ignoreAbility','forceSwitch','selfSwitch','overrideOffensiveStat','overrideDefensiveStat','overrideOffensivePokemon','thawsTarget','stallingMove','noSketch','secondaries','hasCrashDamage','mindBlownRecoil','struggleRecoil','selfdestruct','damage','basePowerCallback'];
const pick=(o,keys)=>{const r={};for(const k of keys){if(o[k]!==undefined&&typeof o[k]!=='function')r[k]=o[k];}return r;};
const miss=JSON.parse(fs.readFileSync('missing.json'));
const out={species:{},moves:{}};
for(const id of miss.species){const s=d.species.get(id);out.species[id]=pick(s,spKeys);}
for(const id of miss.moves){const m=d.moves.get(id);out.moves[id]=pick(m,mvKeys);}
fs.writeFileSync('patch_data.json',JSON.stringify(out));
console.log(Object.keys(out.species).length,Object.keys(out.moves).length);

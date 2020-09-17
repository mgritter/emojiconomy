# Emojiconomy

🌸🍇🍈🍉🍊🍋

A project for Procjam 2019.

This program uses a graph grammar to generate diagrams of the flow of goods
in a toy economy. Plants are eaten, or processed into food and materials.
Ore is mined, refined, and made into machines.

## Goals

 - [X] Generate a graph with all the components of the economy labelled with emoji.
 - [X] Calculate max-flow given a restricted amount of source capacity.
 - [X] Generate a utility function that rewards consumption of multiple types of goods. (Maybe label some as substitutes or complements in the graph.) Figure out how to calculate a flow that maximizes this utility.
 - [X] Duplicate the economic graph and disable parts of it to generate isolated economies that would benefit from trade.
 - [X] Create trading routes between the economies.
 - [ ] Cute web page that animates the result.
  
## To Run

Emojiconomy uses pipenv.  Run

```
pipenv install
pipenv run python -m emojiconomy.planets
```

The files `econ-full-flow.svg` and `econ-full.svg` will show the full economy and its optimal solution.  There will be a lot of debugging output as the auction mechanism is run.  The files `step-NN-PPPP-flow.svg` show intermediate steps for planet/region PPPP after NN auction rounds.

When the auction is completed (200 rounds) the final output will be:
  * `econ-PPPP-flow.svg`: planet PPPP's economic flow
  * `planet-trade.svg`: graph showing all trades
  * `galaxy.pickle`: pickled version of the final state
  
## Talks

I will be presnting a lightning talk on Emojiconomy at Roguelike Celebration 2020.
  * [Slides w/ notes](https://github.com/mgritter/emojiconomy/raw/master/roguelike-celebration-slides-and-notes.pdf)
  * Video (after the event)
  

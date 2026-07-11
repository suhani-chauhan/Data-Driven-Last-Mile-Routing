import streamlit as st

st.markdown(
    """## How to Use This

1. **Pick a route** from the dropdown on the Home page — these are real delivery routes from the dataset, ranging from 19 to 119 stops.
2. **Hit Solve.** You'll see two suggested delivery orders on the map: one based on straight-line distance only, one that also factors in patterns learned from real drivers.
3. **Compare the scores below the map** — this shows how closely each suggested order matches what a real driver actually did on that route. Lower is better.
4. **Want to try your own addresses?** Switch to "Build your own route" mode, add a few stops, and hit Optimize. Note: custom routes only get the straight-line version — the real-driver-pattern layer only works for routes already in the historical dataset.

---

### Reading the Map

- **Black home icon** = the depot, where the route starts.
- **Blue line and markers** = the straight-line route.
- **Orange line and markers** = the smart route.
- **Dark dashed line** = the real driver's actual historical path, shown for comparison.
- Only one route's markers and line are shown at a time. Use the **layer toggle box** in the top-right corner of the map to switch between them, or check more than one at once to compare directly.
- Numbered markers show each stop's position in that route's delivery order (1, 2, 3...). When stops are clustered close together, they collapse into a single numbered bubble — **click the bubble or zoom in** to expand it into the individual stops.
- **Click any numbered marker** to see a popup with that stop's code (or address, in "Build your own route" mode) and its position in the sequence.

### Solving Takes a Little While — That's Expected

Each route is solved twice (once for the straight-line version, once for the smart version), and each solve can take up to a minute, since it's running a real optimization search rather than returning a cached answer. A spinner shows which version is currently being solved. Sample routes use a longer time budget than custom routes, since giving the solver more time produces more consistent, reliable results.

### Delivery Progress

Once a route is solved, a **Delivery Progress** tracker appears below the results:

- It shows the **next stop** to deliver to, and which stop number you're on out of the total.
- Click **Mark as Delivered** to advance to the next stop. A progress bar tracks how much of the route is complete.
- Expand **Full stop list** to see every stop in order, with a checkmark next to the ones already marked delivered.
- Once every stop is marked delivered, you'll see a completion message with the route's total travel time.
- Made a mistake or want to start over? Use **Reset progress** to go back to the beginning of the route.
- Sample-route mode tracks progress against the smart route's order (the system's recommended path); custom routes track progress against the one order they were solved with.

### Building Your Own Route

In "Build your own route" mode:

- **Add stops one at a time**, either by typing a real-world address (which gets automatically converted to map coordinates) or by entering latitude/longitude directly if you already have coordinates or the address lookup doesn't find what you're looking for.
- The **first stop you add is treated as the depot** — the route's starting point — and is labeled accordingly in the stop list.
- You need **at least 3 stops** before the "Optimize Route" button appears.
- Made a mistake? Each stop in the list has its own **Remove** button.
- Custom routes are optimized using straight-line distance only — there isn't enough historical driver data for addresses outside the training dataset to power the smart-route penalty, so that comparison isn't available here.
- Distances for custom routes are estimated as straight-line distance at an assumed driving speed, not pulled from a real mapping/traffic service — treat the travel-time numbers as a rough estimate, not a precise prediction.

### A Few Tips

- If an address doesn't geocode successfully, try being more specific (add a city and state, or a ZIP code), or switch to manual latitude/longitude entry.
- The **score** shown for sample routes only exists because we have a real driver's historical route to compare against — custom routes don't have a score for this reason, only a total estimated travel time.
- Switching modes or routes doesn't lose your progress on the *other* mode — sample-route and custom-route delivery progress are tracked separately."""
)

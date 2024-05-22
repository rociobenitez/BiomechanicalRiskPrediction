import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Declarar las dimensiones y márgenes del gráfico.
const width = 960;
const height = 650;
const marginTop = 60;
const marginTopBar = 120;
const marginRight = 50;
const marginBottom = 50;
const marginLeft = 50;

const csvUrl = '/data/dataset_corredores_processed.csv';

let allData;
let color; // Definir la escala de colores global

d3.dsv(";", csvUrl).then(data => {
    data.forEach(d => {
        d.peso = +d.peso;
        d.impact_gs_run = +d.impact_gs_run;
        d.imc = +d.imc;
        d.articulacion = d.articulacion;
    });

    allData = data;

    const uniqueArticulations = [...new Set(data.map(d => d.articulacion))];
    createCheckboxes(uniqueArticulations);
    createWeightRangeControl();
    updateChart(data);
    updateBarChart(data);
}).catch(error => {
    console.error('Error loading or processing data:', error);
});

function createCheckboxes(articulations) {
    const controls = d3.select("#controls");

    articulations.forEach(articulation => {
        const label = controls.append("label").text(articulation);
        label.append("input")
            .attr("type", "checkbox")
            .attr("value", articulation)
            .attr("checked", true)
            .on("change", filterData);
    });
}

function filterData() {
    const selectedArticulations = Array.from(document.querySelectorAll('#controls input[type="checkbox"]:checked'))
        .map(cb => cb.value);

    const filteredData = allData.filter(d => selectedArticulations.includes(d.articulacion));
    updateChart(filteredData);
    updateBarChart(filteredData);
}

function updateChart(data) {
    d3.select("#chart").selectAll("*").remove();

    // Declarar la escala x (posición horizontal) incluyendo 0 en el dominio.
    const x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.imc)])
        .range([marginLeft, width - marginRight]);

    // Declarar la escala y (posición vertical) incluyendo 0 en el dominio.
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.impact_gs_run)])
        .range([height - marginBottom, marginTop]);

    // Escala de color para la patología afectada.
    color = d3.scaleOrdinal(d3.schemeObservable10)
        .domain([...new Set(data.map(d => d.articulacion))]);

    // Crear el contenedor SVG.
    const svg = d3.select("#chart").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    const g = svg.append("g");

    // Añadir los puntos de dispersión.
    g.append("g")
        .selectAll("circle")
        .data(data)
        .join("circle")
        .attr("cx", d => x(d.imc))
        .attr("cy", d => y(d.impact_gs_run))
        .attr("r", 4)
        .attr("fill", d => color(d.articulacion))
        .on("mouseover", (event, d) => {
            tooltip.transition().duration(200).style("opacity", .95);
            tooltip.html(`<p>IMC: <b>${d.imc}</b><br>Peso: <b>${d.peso} kg</b><br>Impacto: <b>${d.impact_gs_run.toFixed(2)}</b><br>Articulación: <b>${capitalizeFirstLetter(d.articulacion)}</b><br>Sexo: <b>${d.sexo ? 'Masculino' : 'Femenino'}</b></p>`)
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", () => {
            tooltip.transition().duration(500).style("opacity", 0);
        });

    // Añadir la leyenda.
    const legend = svg.append("g")
        .attr("transform", `translate(${width - marginRight - 100}, ${marginTop})`);

    const uniqueArticulations = [...new Set(data.map(d => d.articulacion))];

    legend.selectAll("circle")
        .data(uniqueArticulations)
        .join("circle")
        .attr("cx", 0)
        .attr("cy", (d, i) => i * 20)
        .attr("r", 5)
        .attr("fill", d => color(d));

    legend.selectAll("text")
        .data(uniqueArticulations)
        .join("text")
        .attr("x", 10)
        .attr("y", (d, i) => i * 20 + 5)
        .text(d => capitalizeFirstLetter(d));

    // Añadir el título del gráfico.
    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", marginTop / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "18px")
        .style("font-family", "sans-serif")
        .style("font-weight", "500")
        .text("Impacto durante la carrera vs IMC por Articulación Afectada");

    // Añadir el eje x y su etiqueta.
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0))
        .call(g => g.append("text")
            .attr("x", width)
            .attr("y", marginBottom - 4)
            .attr("fill", "#637381")
            .attr("text-anchor", "end")
            .text("IMC →"))
        .selectAll(".domain, .tick line")
        .attr("stroke", "#637381");

    // Añadir el eje y y su etiqueta, y eliminar la línea de dominio.
    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(y).ticks(height / 40))
        .call(g => g.select(".domain").remove())
        .call(g => g.append("text")
            .attr("x", -marginLeft)
            .attr("y", marginTop)
            .attr("fill", "#637381")
            .attr("text-anchor", "start")
            .text("↑ Impacto (g)"))
        .selectAll(".domain, .tick line")
        .attr("stroke", "#637381");

    // Añadir el contenedor del tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    zoom(svg);
}


function updateBarChart(data) {
    d3.select("#bar-chart").selectAll("*").remove();

    const avgImpactByArticulation = d3.rollups(data, v => ({
        impactoPromedio: d3.mean(v, d => d.impact_gs_run),
        imcPromedio: d3.mean(v, d => d.imc)
    }), d => d.articulacion).map(([key, value]) => ({
        articulacion: key,
        impactoPromedio: value.impactoPromedio,
        imcPromedio: value.imcPromedio
    }));

    const x0 = d3.scaleBand()
        .domain(avgImpactByArticulation.map(d => d.articulacion))
        .range([marginLeft, width - marginRight])
        .padding(0.1);

    const x1 = d3.scaleBand()
        .domain(["Impacto", "IMC"])
        .range([0, x0.bandwidth()])
        .padding(0.05);

    // Establecer valores fijos para los dominios de las escalas y0 y y1
    const y0Max = 10; // Valor máximo fijo para la escala y0 (Impacto)
    const y1Max = 30; // Valor máximo fijo para la escala y1 (IMC)

    const y0 = d3.scaleLinear()
        .domain([0, y0Max]) // Dominio fijo para y0
        .range([height - marginBottom, marginTopBar]);

    const y1 = d3.scaleLinear()
        .domain([0, y1Max]) // Dominio fijo para y1
        .range([height - marginBottom, marginTopBar]);

    const svg = d3.select("#bar-chart").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    const barGroups = svg.append("g")
        .selectAll("g")
        .data(avgImpactByArticulation)
        .join("g")
        .attr("transform", d => `translate(${x0(d.articulacion)},0)`);

    barGroups.selectAll("rect")
        .data(d => [
            { key: "Impacto", value: d.impactoPromedio, articulacion: d.articulacion },
            { key: "IMC", value: d.imcPromedio, articulacion: d.articulacion }
        ])
        .join("rect")
        .attr("x", d => x1(d.key))
        .attr("y", d => d.key === "Impacto" ? y0(d.value) : y1(d.value))
        .attr("width", x1.bandwidth())
        .attr("height", d => d.key === "Impacto" ? y0(0) - y0(d.value) : y1(0) - y1(d.value))
        .attr("fill", d => color(d.articulacion))
        .attr("opacity", d => d.key === "IMC" ? 0.7 : 1)
        .on("mouseover", function (event, d) {
            d3.select(this).attr("opacity", 1).attr("stroke", "#000").attr("stroke-width", 1.5);
            barGroups.selectAll("rect").filter(x => x !== d).attr("opacity", 0.3);
            tooltip.transition().duration(200).style("opacity", .95);
            tooltip.html(`
                    <p>${d.key}: <b>${d.value.toFixed(2)}</b><br>
                    Articulación: <b>${capitalizeFirstLetter(d.articulacion)}</b><br>
                    Prom. Edad: <b>${d3.mean(allData.filter(item => item.articulacion === d.articulacion), item => item.edad).toFixed(2)}</b><br>
                    Prom. Altura: <b>${d3.mean(allData.filter(item => item.articulacion === d.articulacion), item => item.altura).toFixed(2)} cm</b><br>
                    Prom. Peso: <b>${d3.mean(allData.filter(item => item.articulacion === d.articulacion), item => item.peso).toFixed(2)} kg</b><br>
                `)
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function () {
            d3.select(this).attr("opacity", d => d.key === "IMC" ? 0.7 : 1).attr("stroke", "none");
            barGroups.selectAll("rect").attr("opacity", d => d.key === "IMC" ? 0.7 : 1);
            tooltip.transition().duration(500).style("opacity", 0);
        });

    // Añadir el contenedor del tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    // Añadir los valores promedio dentro de cada barra.
    barGroups.selectAll("text")
        .data(d => [
            { key: "Impacto", value: d.impactoPromedio, y: y0(d.impactoPromedio), articulacion: d.articulacion },
            { key: "IMC", value: d.imcPromedio, y: y1(d.imcPromedio), articulacion: d.articulacion }
        ])
        .join("text")
        .attr("x", d => x1(d.key) + x1.bandwidth() / 2)
        .attr("y", d => d.y - 5)
        .attr("text-anchor", "middle")
        .text(d => d.value.toFixed(2))
        .style("fill", "#000")
        .style("font-size", "14px");

    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(x0).tickSizeOuter(0))
        .call(g => g.selectAll(".domain, .tick line").attr("stroke", "#637381"));

    svg.append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(d3.axisLeft(y0).ticks(height / 40))
        .call(g => g.selectAll(".domain, .tick line").attr("stroke", "#637381"))
        .call(g => g.selectAll("text").style("font-size", "13px"))
        .call(g => g.append("text")
            .attr("x", -marginLeft + 5)
            .attr("y", 95)
            .attr("fill", "#637381")
            .attr("text-anchor", "start")
            .text("↑ Impacto (g)"));

    svg.append("g")
        .attr("transform", `translate(${width - marginRight},0)`)
        .call(d3.axisRight(y1).ticks(height / 40))
        .call(g => g.selectAll(".domain, .tick line").attr("stroke", "#637381"))
        .call(g => g.selectAll("text").style("font-size", "13px"))
        .call(g => g.append("text")
            .attr("x", 30)
            .attr("y", 95)
            .attr("fill", "#637381")
            .attr("text-anchor", "end")
            .text("IMC ↑"));

    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", marginTopBar / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "18px")
        .style("font-family", "sans-serif")
        .style("font-weight", "500")
        .text("Promedio de Impacto y IMC por Articulación Afectada");
}


function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function zoom(svg) {
    const extent = [[marginLeft, marginTop], [width - marginRight, height - marginBottom]];

    svg.call(d3.zoom()
        .scaleExtent([1, 8])
        .translateExtent(extent)
        .extent(extent)
        .on("zoom", zoomed));

    function zoomed(event) {
        const { transform } = event;
        const g = svg.select("g");
        g.attr("transform", transform);
        g.selectAll("circle").attr("cx", d => transform.applyX(x(d.imc))).attr("cy", d => transform.applyY(y(d.impact_gs_run)));
        svg.select(".x-axis").call(d3.axisBottom(x).scale(transform.rescaleX(x)));
        svg.select(".y-axis").call(d3.axisLeft(y).scale(transform.rescaleY(y)));
    }
}

// Añadir el control de rango para el peso
function createWeightRangeControl() {
    const controlContainer = d3.select("#weight-range-container")

    controlContainer.append("label")
        .attr("for", "weight-range")
        .text("Rango de Peso: ");

    const minWeight = d3.min(allData, d => d.peso);
    const maxWeight = d3.max(allData, d => d.peso);

    controlContainer.append("input")
        .attr("type", "range")
        .attr("id", "weight-range")
        .attr("min", minWeight)
        .attr("max", maxWeight)
        .attr("step", 1)
        .attr("value", maxWeight)
        .on("input", filterDataByWeight);

    controlContainer.append("span")
        .attr("id", "weight-range-value")
        .text(maxWeight);
}

function filterDataByWeight() {
    const weight = +d3.select("#weight-range").property("value");
    d3.select("#weight-range-value").text(weight);

    const filteredData = allData.filter(d => d.peso <= weight);
    updateBarChart(filteredData);
}

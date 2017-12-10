
function predict() {

    console.log("building ajax request...");

    // Grab value from project description textarea
    descElem = $('#project-desc');
    var projectDesc = descElem[0].innerText;

    //console.log(projectDesc);

    // Create and send ajax request
    var xhttp = new XMLHttpRequest();
    xhttp.open("POST", "/predict", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("d=" + encodeURIComponent(projectDesc));

    console.log("sending ajax request...");

    xhttp.onload = function() {
        highlight(projectDesc, this.responseText);
    };

    console.log("ajax callback created");

}

function highlight(text, valuesJson) {

    console.log("parsing ajax response...");
    //console.log(valuesJson);

    var json = JSON.parse(valuesJson);
    //console.log(json);
    var values = json.values;
    var score = json.score;
    //console.log(json.score);

    score = Math.floor(score * 100);


    $('#score').html(score + "%");

    var words = text.split(" ");

    var output = "";

    console.log("highlighting words...");

    for (var i = 0; i < words.length; i++) {

        output = output + wrap(words[i], values[i]) + " ";

    }

    console.log("highlight complete");

    $('#project-desc').html(output);

}

/**
 * Wraps the given word into a span with a style color: attribute corresponding
 * to the given value.
 */
function wrap(word, value) {

    var red = Math.floor(255*value);
    var green = Math.floor(255*value);
    var blue = Math.floor(255*value);

    return '<span style="display: inline-block; color: rgba(' + 0 + ', ' + green + ', ' + 0 + ', 1) !important;">' + word + '</span>';

}

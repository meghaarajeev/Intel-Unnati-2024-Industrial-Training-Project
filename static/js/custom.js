$(document).ready(function() {

    // Sticky header
    $(window).scroll(function() {
        if ($(this).scrollTop() > 50) {  
            $('header').addClass("sticky");
        } else {
            $('header').removeClass("sticky");
        }
    });

    // Copyrights Year Auto-update
    function newDate() {
        return new Date().getFullYear();
    }
    document.onload = document.getElementById("autodate").innerHTML = newDate();

    // Show validation form and scroll to it
    $('#validate-btn').on('click', function() {
        $('#validate').show();
        $('html, body').animate({
            scrollTop: $("#validate").offset().top
        }, 1000);
    });

    // Form submission for prediction
    $('#predict-form').on('submit', function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                // Redirect to result page with the prediction result
                window.location.href = "/result?template_name=" + encodeURIComponent(data.template_name) + "&validation_message=" + encodeURIComponent(data.validation_message);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                $('#prediction-result').html('<p class="text-danger">Error: ' + errorThrown + '</p>');
            }
        });
    });
});

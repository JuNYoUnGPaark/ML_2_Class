// for (let i = 0; i < tab_ctn; i++) {
//         btn.eq(i).on('click', function() {
//         $('.orange').removeClass('orange');
//         btn.eq(i).addClass('orange');
//         ctn.removeClass('show');
//         ctn.eq(i).addClass('show');
//     });
// }

var btn = $('.tab-button');
var ctn = $('.tab-content');
var tab_ctn = $('.tab-button').length;

function tabOpen(i) {
    $('.orange').removeClass('orange');
    btn.eq(i).addClass('orange');
    ctn.removeClass('show');
    ctn.eq(i).addClass('show');
}

$('.list').click(function(e) {
    tabOpen(e.target.dataset.id);
});
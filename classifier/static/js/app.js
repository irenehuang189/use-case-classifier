$(document).ready(function(){

  var $container = $('.masonry-grid');
  $container.imagesLoaded(function(){
    $container.masonry({
      itemSelector: '.grid-item',
      columnWidth: '.grid-item',
      horizontalOrder: true,
      transitionDuration: '0.8s',
      stagger: 30
    });
  });

  var $animation_elements = $('.grid-item');
  var $window = $(window);

  function check_if_in_view() {
    var window_height = $window.height();
    var window_top_position = $window.scrollTop();
    var window_bottom_position = (window_top_position + window_height);
   
    $.each($animation_elements, function() {
      var $element = $(this);
      var element_height = $element.outerHeight();
      var element_top_position = $element.offset().top;
      var element_bottom_position = (element_top_position + element_height);
   
      //check to see if this current container is within viewport
      if ((element_bottom_position >= window_top_position) &&
          (element_top_position <= window_bottom_position)) {
        $element.addClass('animate');
      }
    });
  }

  $window.on('scroll resize', check_if_in_view);
  $window.trigger('scroll');
});

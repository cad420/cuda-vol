#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <cuda/misc.hpp>
#include <cuda/memory.hpp>
#include <utils/attribute.hpp>
#include <utils/concepts.hpp>

namespace vol
{
namespace _
{
template <typename Voxel>
struct VolumeInner : NoCopy, NoMove
{
	VolumeInner( std::ifstream &&_, cuda::Extent const &block_dim ) :
	  _( std::move( _ ) ),
	  block_dim( block_dim ),
	  block_size( sizeof( Voxel ) * block_dim.size() )
	{
	}

	std::ifstream _;
	cuda::Extent block_dim;
	std::size_t block_size;
	uint3 padding = uint3{ 0, 0, 0 };
};

}  // namespace _

template <typename Voxel>
struct Volume;

template <typename Voxel>
struct ArchievedVolumeBlock;

template <typename Voxel>
struct VolumeBlock
{
private:
	struct Inner : NoCopy, NoMove
	{
		std::string _;
		cuda::Extent dim;
	};

public:
	const Voxel *data() const
	{
		return reinterpret_cast<const Voxel *>( _->_.c_str() );
	}
	cuda::MemoryView3D<Voxel> view() const
	{
		auto view_info = cuda::MemoryView2DInfo{}
						   .set_stride( this->_->dim.width * sizeof( Voxel ) )
						   .set_width( this->_->dim.width )
						   .set_height( this->_->dim.height );
		return cuda::MemoryView3D<Voxel>( const_cast<char *>( _->_.c_str() ),
										  view_info, this->_->dim );
	}

private:
	VolumeBlock( std::string &&_, cuda::Extent dim )
	{
		this->_->_ = std::move( _ );
		this->_->dim = dim;
	}

	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
	friend struct ArchievedVolumeBlock<Voxel>;
};

template <typename Voxel>
struct ArchievedVolumeBlock
{
	VolumeBlock<Voxel> unarchieve() const
	{
		std::string buffer;
		buffer.resize( _->block_size );
		auto buffer_ptr = const_cast<char *>( buffer.c_str() );
		auto nread = _->_.seekg( offset )
					   .read( buffer_ptr, _->block_size )
					   .gcount();
		if ( nread != _->block_size ) {
			throw std::runtime_error( "failed to read block" );
		}
		return VolumeBlock<Voxel>( std::move( buffer ), _->block_dim );
	}
	uint3 index() const { return idx; }

private:
	std::shared_ptr<_::VolumeInner<Voxel>> _;
	std::size_t offset;
	uint3 idx;
	friend struct Volume<Voxel>;
};

namespace lvd
{
struct Header
{
	uint32_t magic_number;
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t log_block_dim;
	uint32_t padding;
	uint32_t original_width;
	uint32_t original_height;
	uint32_t original_depth;
};
}

template <typename Voxel>
struct Volume
{
	static Volume from_raw( const std::string &file_name, cuda::Extent const &block_dim )
	{
		Volume vol;
		vol._ = std::make_shared<_::VolumeInner<Voxel>>(
		  std::ifstream( file_name, std::ios::in | std::ios::binary ),
		  block_dim );
		vol.block_stride = vol._->block_size;
		vol.raw_all = vol.all = block_dim;
		return vol;
	}
	// fake implementation read one raw file ant make a 2x2x2 grid
	static Volume from_lvd( const std::string &file_name )
	{
		Volume vol;
		lvd::Header header;
		std::ifstream is( file_name, std::ios::in | std::ios::binary );
		is.read( reinterpret_cast<char *>( &header ), sizeof( header ) );
		// std::cout << header.width << std::endl
		// 		  << header.height << std::endl
		// 		  << header.depth << std::endl
		// 		  << header.log_block_dim << std::endl
		// 		  << header.padding << std::endl
		// 		  << header.original_width << std::endl
		// 		  << header.original_height << std::endl
		// 		  << header.original_height << std::endl;
		auto len = 1 << header.log_block_dim;
		auto block_dim = cuda::Extent{}
						   .set_width( len )
						   .set_height( len )
						   .set_depth( len );
		vol._ = std::make_shared<_::VolumeInner<Voxel>>( std::move( is ), block_dim );
		vol._->padding = uint3{ header.padding, header.padding, header.padding };
		vol.offset = sizeof( header );
		// vol.block_stride = 0;
		vol.block_stride = vol._->block_size;
		vol.grid_dim = dim3( header.width / len,
							 header.height / len,
							 header.depth / len );
		vol.all = vol.grid_dim * ( len - vol._->padding * 2 );
		vol.raw_all = dim3{ header.original_width,
							header.original_height,
							header.original_depth };
		return vol;
	}

public:
	dim3 dim() const { return grid_dim; }
	cuda::Extent block_dim() const { return _->block_dim; }
	uint3 padding() const { return _->padding; }

	ArchievedVolumeBlock<Voxel> get_block( uint3 idx )
	{
		auto block_id = idx.x +
						idx.y * grid_dim.x +
						idx.z * grid_dim.x * grid_dim.y;
		ArchievedVolumeBlock<Voxel> block;
		block._ = _;
		block.offset = offset + block_id * block_stride + block_offset;
		block.idx = idx;
		return block;
	}
	std::vector<ArchievedVolumeBlock<Voxel>> get_blocks( uint3 idmin, uint3 idmax )
	{
		uint3 idx;
		std::vector<ArchievedVolumeBlock<Voxel>> _;
		for ( idx.z = idmin.z; idx.z < idmax.z; ++idx.z ) {
			for ( idx.y = idmin.y; idx.y < idmax.y; ++idx.y ) {
				for ( idx.x = idmin.x; idx.x < idmax.x; ++idx.x ) {
					_.emplace_back( get_block( idx ) );
				}
			}
		}
		return _;
	}
	std::vector<ArchievedVolumeBlock<Voxel>> get_blocks()
	{
		return get_blocks( uint3{ 0, 0, 0 }, uint3{ grid_dim.x, grid_dim.y, grid_dim.z } );
	}
	float scale() const
	{
		return std::min( std::min( float( all.x ) / raw_all.x,
								   float( all.y ) / raw_all.y ),
						 float( all.z ) / raw_all.z );
	}

private:
	std::shared_ptr<_::VolumeInner<Voxel>> _;
	dim3 grid_dim = dim3( 1, 1, 1 );
	std::size_t offset = 0;
	std::size_t block_stride = 0;
	std::size_t block_offset = 0;
	dim3 all, raw_all;
};

}  // namespace vol
